# Building jaxlib for linux Arm64 to run in Docker on Apple Silicon

Run on Apple Silicon as its much faster.

## Build the dockerfile

```
$ docker build -f build_jax.dockerfile . -t build_jax_arm64"
```

## Run the container to start the build

```
$ mkdir bazel_cache
$ docker run -it -v `pwd`/bazel_cache:/root/.cache build_jax_arm64 bash
```

## Verify you are running arm64

```
$ uname -a
Linux 918808416e70 5.15.49-linuxkit-pr #1 SMP PREEMPT Thu May 25 07:27:39 UTC 2023 aarch64 GNU/Linux
```

## Confirm your GLIBC version

If this doesn't match the environment you will be running the jaxlib wheel it might fail.
In this case we are using the same `FROM python:3.11.4-slim as build` to build as jaxlib we do in our production container to run it.

```
$ ldd --version
ldd (Debian GLIBC 2.36-9+deb12u1) 2.36
```

## Checkout the version you want to build

```
$ cd jax
$ git fetch --all --tags
$ git checkout tags/jaxlib-v0.4.14 -b jaxlib-v0.4.14
```

## Start the build

```
$ python3 /jax/build/build.py --target_cpu_features native
```

## Check it worked

```
$ pip install dist/*.whl
```

## Copy out the wheel

```
$ pip install --user magic-wormhole
$ /root/.local/bin/wormhole send /jax/build/dist/jaxlib-0.4.14-cp311-none-manylinux2014_aarch64.whl
# run wormhole recieve on the other end
```
