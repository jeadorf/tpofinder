## About tpofinder ##
An experimental detector for textured planar objects. It first matches local features from the scene to the models, and then estimates homographies by random sample consensus.

![http://tpofinder.googlecode.com/git/doc/screenshot-1.png](http://tpofinder.googlecode.com/git/doc/screenshot-1.png)

![http://tpofinder.googlecode.com/git/doc/screenshot-2.png](http://tpofinder.googlecode.com/git/doc/screenshot-2.png)

## Building tpofinder ##

tpofinder uses CMake as build system. OpenCV 2.4 is required.

```
    git clone https://code.google.com/p/tpofinder/
    cd tpofinder
    mkdir build
    cd build
    cmake ..
    make
```

## Running tpofinder ##

You can use your webcam to feed images to tpofinder. Some object models have already been included in the `data` directory: show one of these objects (or a printout) to the camera and see whether it is recognized.

```
    tpofind --webcam
```

It is also possible to detect objects on images given as path names on the command-line:

```
    tpofind --file some-image.jpg
```

If you have a directory full of images, you can pass them to tpofind via standard input:

```
    cd tpofinder
    find some-folder -iname "*.jpg" -type f | tpofind
```

## Testing tpofinder ##

There are a couple of unit tests. Run them from the root of the source directory as follows:

```
    utest
```

You can also specify further options to the Google test runner, see:

```
    utest --help
```