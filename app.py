from celery import uuid
from duckling import DucklingWrapper
import fastText
from flair.data import Sentence
from flair.models import SequenceTagger
from flashtext import KeywordProcessor
from flask import Flask, jsonify, request
from pathlib import Path
from postal.parser import parse_address
import re
from shutil import move
from tasks import make_celery
import time
from typing import Any, Dict, List, Tuple

ENTITY_DATE = 'date'
ENTITY_NUMBER = 'number'
ENTITY_CURRENCY = 'currency'
ENTITY_ADDRESS = 'address'
ENTITY_PERSON = 'person'

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)
model = None
entity_reverse_lookup: Dict[str, str] = None
regexprs: Dict[str, list] = None
keyword_processor: KeywordProcessor = None
d = None
tagger = None
duckling_entities = {ENTITY_DATE, ENTITY_NUMBER, ENTITY_CURRENCY}
tagger_entities = {ENTITY_ADDRESS, ENTITY_PERSON}
available_system_entities = {ENTITY_DATE, ENTITY_NUMBER, ENTITY_CURRENCY, ENTITY_ADDRESS, ENTITY_PERSON}
system_entities = {ENTITY_DATE, ENTITY_NUMBER, ENTITY_CURRENCY, ENTITY_ADDRESS, ENTITY_PERSON}
jobs = {}


def init():
    global model, entity_reverse_lookup, regexprs, keyword_processor, d, tagger
    root_path = Path(__file__).parent
    model_path = str(root_path / 'models/classifier.ftz')
    model = fastText.load_model(model_path)
    entities_path = str(root_path / 'models/entities.csv')
    entity_reverse_lookup, synonyms, regexprs = load_entities(entities_path)
    keyword_processor = prepare_keyword_processor(synonyms)
    if len(duckling_entities.intersection(system_entities)) > 0:
        d = DucklingWrapper()

    if len(tagger_entities.intersection(system_entities)) > 0:
        tagger = SequenceTagger.load('ner')


@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/set-system-entity/<entity>', methods=['POST'])
def set_system_entity(entity):
    global d, tagger
    if entity in available_system_entities:
        if entity not in system_entities:
            system_entities.add(entity)
            if entity in duckling_entities and d is None:
                d = DucklingWrapper()

            if entity in tagger_entities and tagger is None:
                tagger = SequenceTagger.load('ner')

        return 'done', 200

    else:
        return '{} not available'.format(entity), 404


@app.route('/unset-system-entity/<entity>', methods=['POST'])
def unset_system_entity(entity):
    if entity in available_system_entities:
        if entity in system_entities:
            system_entities.remove(entity)

        return 'done', 200

    else:
        return '{} not available'.format(entity), 404


@app.route('/intent', methods=['POST', 'PUT'])
def intent():
    content = request.json
    return content


@app.route('/train', methods=['POST'])
def train():
    task_id = uuid()
    train_model.apply_async(link=handle_training_done.s(task_id=task_id))
    jobs[task_id] = 'training'
    return task_id, 200


@app.route('/status/<task_id>')
def get_training_status(task_id):
    if task_id in jobs:
        status = jobs[task_id]
        return status, 200
    else:
        return 'Task ({}) not found'.format(task_id), 404


@app.route('/message', methods=['POST'])
def message():
    content = request.json
    utter = content['input']['text']
    label = model.predict(utter)
    intent_ = label[0][0][9:]
    confidence = label[1][0]
    entities = []
    keywords_found = keyword_processor.extract_keywords(utter, span_info=True)
    for keyword in keywords_found:
        entities.append({
            'entity': entity_reverse_lookup[keyword[0]],
            'location': keyword[1:],
            'value': keyword[0],
            'confidence': 1.0
        })

    matches = match_regexprs(utter, regexprs)
    for match in matches:
        match['entity'] = entity_reverse_lookup[match['value']]

    entities.extend(matches)
    entities.extend(match_system_entities(utter))
    resp = {
        'output': {
            'generic': [
                {
                    'response_type': 'text',
                    'text': utter
                }
            ],
            'intents': [
                {
                    'intent': intent_,
                    'confidence': confidence
                }
            ],
            'entities': entities
        }
    }
    return jsonify(resp)


def load_entities(file_path: str) -> Tuple[Dict[str, str], Dict[str, list], Dict[str, list]]:
    entity_reverse_lookup_ = {}
    synonyms_ = {}
    regexprs_ = {}
    with open(file_path, 'r') as f:
        for line in f:
            values = line.strip().split(',')
            if len(values) > 1:  # otherwise entity specification incomplete
                entity_name = values[0]
                entity_value = values[1]
                entity_reverse_lookup_[entity_value] = entity_name
                if len(values) > 2 and values[2].startswith('/'):
                    # A regular expr
                    regexprs_[entity_value] = [expr[1:-1] for expr in values[2:]]  # strip '/.../' markers
                else:
                    # A synonym
                    synonyms_[entity_value] = values[1:]  # include the entity_value

    return entity_reverse_lookup_, synonyms_, regexprs_


def match_regexprs(utter: str, regexprs_: Dict[str, list]) -> List[Dict[str, Any]]:
    matches = []
    for entity_value, exprs in regexprs_.items():
        for expr in exprs:
            for match in re.finditer(expr, utter):
                groups = [{
                    'group': 'group_0',
                    'location': list(match.span())
                }]
                for i, g in enumerate(match.groups()):
                    groups.append({
                        'group': 'group_{}'.format(i + 1),
                        'location': list(match.span(i + 1))
                    })

                entity = {
                    'location': list(match.span()),
                    'value': entity_value,
                    'confidence': 1.0,
                    'groups': groups
                }
                matches.append(entity)

    return matches


def match_system_entities(utter):
    matches = []
    if ENTITY_DATE in system_entities:
        results = d.parse_time(utter)
        for result in results:
            matches.append({
                'entity': 'sys-date',
                'location': [result['start'], result['end']],
                'value': result['value']['value'],
                'confidence': 1.0
            })

    if ENTITY_NUMBER in system_entities:
        results = d.parse_number(utter)
        for result in results:
            matches.append({
                'entity': 'sys-number',
                'location': [result['start'], result['end']],
                'value': result['value']['value'],
                'confidence': 1.0
            })

    if ENTITY_CURRENCY in system_entities:
        results = d.parse_money(utter)
        for result in results:
            matches.append({
                'entity': 'sys-currency',
                'location': [result['start'], result['end']],
                'value': result['value']['value'],
                'confidence': 1.0
            })

    sentence = None

    if ENTITY_ADDRESS in system_entities:
        sentence = Sentence(utter)
        tagger.predict(sentence)
        has_location = False
        for entity in sentence.get_spans('ner'):
            if entity.tag == 'LOC':
                has_location = True
                break

        parts = parse_address(utter, language='en', country='AU')
        if not (len(parts) == 1 and parts[0][1] == 'house' and not has_location):
            metadata = {label: value for value, label in parts}
            matches.append({
                'entity': 'sys-location',
                'location': [0, len(utter)],
                'value': normalize_address(metadata),
                'confidence': 1.0,
                'metadata': metadata
            })

    if ENTITY_PERSON in system_entities:
        if sentence is None:
            sentence = Sentence(utter)
            tagger.predict(sentence)

        for entity in sentence.get_spans('ner'):
            if entity.tag == 'PER':
                matches.append({
                    'entity': 'sys-person',
                    'location': [entity.start_pos, entity.end_pos],
                    'value': entity.text,
                    'confidence': entity.score
                })

    return matches


def prepare_keyword_processor(synonyms_: Dict[str, list]) -> KeywordProcessor:
    """
    28x faster than a compiled regexp for 1,000 keywords
    https://github.com/vi3k6i5/flashtext

    :param synonyms_: dict of entity synonyms
    :return:
    """
    kp = KeywordProcessor(case_sensitive=True)
    kp.add_keywords_from_dict(synonyms_)
    return kp


def normalize_address(parts: Dict) -> str:
    unit = None
    if 'unit' in parts:
        unit = 'Unit ' + parts['unit']
    elif 'level' in parts:
        unit = 'Level ' + parts['level']

    addr = [
        unit,
        parts.get('house_number', None),
        parts.get('road', None),
        parts.get('suburb', None),
        parts.get('city', None),
        parts.get('state', None),
        parts.get('postcode', None)
    ]
    return ' '.join(x for x in addr if x is not None)


@celery.task()
def train_model():
    root_path = Path(__file__).parent
    data_path = root_path / 'models/intents.txt'
    # model_bin_path = root_path / 'models/classifier.bin'
    quantized_path = root_path / 'models/classifier.ftz'
    ts = time.strftime('%Y%m%d%H%M%S')

    # Backup old model if exists
    # if model_bin_path.exists():
    #     move(model_bin_path, root_path / 'models/classifier_{}.bin'.format(ts))

    if quantized_path.exists():
        move(quantized_path, root_path / 'models/classifier_{}.ftz'.format(ts))

    model_ = fastText.train_supervised(str(data_path), lr=1.0, epoch=25, wordNgrams=3, loss='hs')
    # model_.save_model(str(model_bin_path))
    model_.quantize(input=str(data_path), cutoff=100000, retrain=True, qnorm=True)
    model_.save_model(str(quantized_path))


# noinspection PyUnusedLocal
@celery.task()
def handle_training_done(result, task_id=None):
    # result is None as `train_model` doesn't return anything
    # TODO - `jobs` not updated in main thread
    jobs[task_id] = 'done'


init()


if __name__ == '__main__':
    app.run()
