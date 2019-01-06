fastintent
==========

Intent and entity extraction API.


Installing libpostal
--------------------

On Ubuntu/Debian::

    sudo apt-get install curl autoconf automake libtool python-dev pkg-config

On CentOS/RHEL::

    sudo yum install curl autoconf automake libtool python-devel pkgconfig

On Mac OSX::

    brew install curl autoconf automake libtool pkg-config

Installing libpostal::

    git clone https://github.com/openvenues/libpostal
    cd libpostal
    ./bootstrap.sh
    ./configure --datadir=[...some dir with a few GB of space...]
    make
    sudo make install

    # On Linux it's probably a good idea to run
    sudo ldconfig

Note: Gets hung up on "Old version of datadir detected, removing...". Running `make` again
seems to work.

If you get the error "Error loading transliteration module, dir=(null)", try::

    make distclean

and trying the above instructions again. If the data (about 1.8GB) successfully downloaded
the first time, add `--disable-data-download` to the `configure` command. Note that the
data is put into a sub-directory "libpostal" under the `datadir`.
