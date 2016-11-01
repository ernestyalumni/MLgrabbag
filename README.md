# MLgrabbag
MLgrabbag - Machine Learning grab bag: includes (pedagogical) examples and implementations of Machine Learning; notes

## Abridged Table of Contents (i.e. Doesn't contain everything)

- Notes on `theano`
- Installation of NVIDIA CUDA on Fedora 23 Workstation (Linux)
  * Recovering from disastrous `dnf update` that adds a new kernel and trashes video output

## Notes on `theano`

| filename | directory | Description |
| :-------- | :---------: | -----------: |
| tutorial_theano.ipynb | `./` | jupyter notebook based on Theano's documentation tutorial |
| supervised-theano.ipynb | `./` | Further explorations of *supervised learning* with `theano`, in a jupyter notebook |
| `simple_logreg.py` | `./` | Simple logistic regression, from Theano's documentation's tutorial |

- `simple_logreg.py` - from Theano's documentation and its tutorial, to run this on the GPU, I used the following:



```
THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python simple_logreg.py
```



# Installation of NVIDIA CUDA on Fedora 23 Workstation (Linux)


Installation of NVIDIA’s [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) on a Fedora 23 Workstation was nontrivial; part of the reason is that it appears that *7.5* is the latest version of the CUDA Toolkit (as of 20150512), and 7.5 only supports (for sure) Fedora 21.  And, this 7.5 version supports (out of the box) C compiler gcc up to version `4.*` and not gcc 5.  But there’s no reason why the later versions, Fedora 23 as opposed to Fedora 21, gcc 5 vs. gcc `4.*`, cannot be used (because I got CUDA to work on my setup, including samples).  But I found that I had to make some nontrivial symbolic linking (`ln`).  

I wanted to install CUDA for Udacity’s [Intro to Parallel Programming](https://www.udacity.com/course/intro-to-parallel-programming--cs344), and in particular, in the very first lesson or video, [Intro to the Class](https://classroom.udacity.com/courses/cs344/lessons/55120467/concepts/658304810923), for instructions on running CUDA locally, only the links to the official NVIDIA documentation were given, in particular for Linux,   
http://docs.nvidia.com/cuda/cuda-getting-started-guide-for-linux/index.html  
But one only needs to do a Google search and read some forum posts that installing CUDA, Windows, Mac, or Linux, is highly nontrivial.  

I’ll point out how I did it, and refer to the links that helped me (sometimes you simply follow, to the letter, the instructions there) and other links in which you should follow the instructions, but modify to suit your (my) system, and what *NOT* to do (from my experience).  

## Gist, short summary, steps to do (without full details), to just get CUDA to work (no graphics)

My install procedure assumes you are using the latest proprietary NVIDIA Accelerated Graphics Drivers for Linux.  I removed and/or blacklisted any other open-source versions of nvidia drivers, and in particular blacklisted nouveau.  See my [blog post](https://ernestyalumni.wordpress.com/#OhNoFedoraetNvidia) for details and description.  

* Download the latest [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) (appears to be 7.5 as of 20160512).  For my setup, I clicked on the boxes Linux for Operation System, x86_64 for Architecture, Fedora for Distribution, 21 for Version (only one there), runfile (local) for Installer Type (it was the first option that appeared).  Then I modified the instructions on their webpage: 

	1. Run `sudo sh cuda_7.5.18_linux.run`
	2. Follow the command-line prompts.

Instead, I did

```
$ sudo sh cuda_7.5.18_linux.run --override
```
with the `- -override` flag to use gcc 5 so I **did not** have to downgrade to gcc `4.*`.  

Here is how I selected my options at the command-line prompts (and part of the result):

```
$ sudo sh cuda_7.5.18_linux.run --override

-------------------------------------------------------------
Do you accept the previously read EULA? (accept/decline/quit): accept
You are attempting to install on an unsupported configuration. Do you wish to continue? ((y)es/(n)o) [ default is no ]: yes
Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 352.39? ((y)es/(n)o/(q)uit): n 
Install the CUDA 7.5 Toolkit? ((y)es/(n)o/(q)uit): y
Enter Toolkit Location [ default is /usr/local/cuda-7.5 ]: 
Do you want to install a symbolic link at /usr/local/cuda? ((y)es/(n)o/(q)uit): y
Install the CUDA 7.5 Samples? ((y)es/(n)o/(q)uit): y
Enter CUDA Samples Location [ default is /home/[yournamehere] ]: /home/[yournamehere]/Public
Installing the CUDA Toolkit in /usr/local/cuda-7.5 ...
Missing recommended library: libGLU.so
Missing recommended library: libX11.so
Missing recommended library: libXi.so
Missing recommended library: libXmu.so

Installing the CUDA Samples in /home/[yournamehere]/ ...
Copying samples to /home/propdev/Public/NVIDIA_CUDA-7.5_Samples now...
Finished copying samples.
```

Again, Fedora 23 was not a supported configuration, but I wished to continue.  I had already installed NVIDIA Accelerated Graphics Driver for Linux (that’s how I was seeing my X graphical environment) but it was a later version **361.* ** and I did not want to uninstall it and then reinstall, which was recommended by other webpages (I had already gone through the [mini-nightmare of reinstalling these drivers before](https://ernestyalumni.wordpress.com/#OhNoFedoraetNvidia), which can trash your X11 environment that you depend on for a functioning GUI).  

* Continuing, this was also printed out by CUDA’s installer: 


Installing the CUDA Samples in /home/propdev/Public ...
Copying samples to /home/propdev/Public/NVIDIA_CUDA-7.5_Samples now...
Finished copying samples.

```
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-7.5
Samples:  Installed in /home/[yournamehere]/Public, but missing recommended libraries

Please make sure that
 -   PATH includes /usr/local/cuda-7.5/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-7.5/lib64, or, add /usr/local/cuda-7.5/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run the uninstall script in /usr/local/cuda-7.5/bin
To uninstall the NVIDIA Driver, run nvidia-uninstall

Please see CUDA_Installation_Guide_Linux.pdf in /usr/local/cuda-7.5/doc/pdf for detailed information on setting up CUDA.

***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 352.00 is required for CUDA 7.5 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run -silent -driver

Logfile is /tmp/cuda_install_7123.log
```

For “ `PATH includes /usr/local/cuda-7.5` ” I do 
```
$ export PATH=/usr/local/cuda-7.5/bin:$PATH
```
as suggested by Chapter 6 of [CUDA_Getting_Started_Linux.pdf](http://developer.download.nvidia.com/compute/cuda/6_5/rel/docs/CUDA_Getting_Started_Linux.pdf)

Dealing with the `LD_LIBRARY_PATH`, I did this: I created a new text file (open up your favorite text editor) in `/etc/ld.so.conf.d` called `cuda.conf`, e.g. I used emacs:
```
sudo emacs cuda.conf
```
and I pasted in the directory
```
/usr/local/cuda-7.5/lib64
```
(since my setup is 64-bit) into this text file.  I did this because my `/etc/ld.so.conf` file includes files from `/etc/ld.so.conf.d`, i.e. it says
```
include ld.so.conf.d/*.conf
```
Make sure this change for `LD_LIBRARY_PATH` is made by running the command
```
ldconfig
```
as root.  


I check the status of this “linking” to `PATH` and `LD_LIBRARY_PATH` with the `echo` command, each time I reboot, or log back in, or start a new Terminal window:
```
echo $PATH
echo $LD_LIBRARY_PATH
```

* Patch the `host_config.h` header file

cf. [Install NVIDIA CUDA on Fedora 22 with gcc 5.1](https://www.pugetsystems.com/labs/articles/Install-NVIDIA-CUDA-on-Fedora-22-with-gcc-5-1-654/) and [CUDA incompatible with my gcc version](http://stackoverflow.com/questions/6622454/cuda-incompatible-with-my-gcc-version)

To use gcc 5 instead of gcc `4.*`, I needed to patch the `host_config.h` header file because I kept receiving errors.  What worked for me was doing this to the file - original version:

```
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9)

#error -- unsupported GNU version! gcc versions later than 4.9 are not supported!

#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9) */
```

Commented-out version (these 3 lines)
```
// #if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9)

// #error -- unsupported GNU version! gcc versions later than 4.9 are not supported!

// #endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ > 9) */
```

Afterwards, I did not have any problems with c compiler gcc incompatibility (yet).  

* At this point CUDA runs without problems if no graphics capabilities are needed.  For instance, as a sanity check, I ran, from the installed samples with CUDA, I made `deviceQuery` and ran it:
```
$ cd ~/NVIDIA_CUDA-7.5_Samples/1_Utilities/deviceQuery
$ make -j12
$ ./deviceQuery
```

And then if your output looks something like this, then **success!**

```
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "GeForce GTX 980 Ti"
  CUDA Driver Version / Runtime Version          8.0 / 7.5
  CUDA Capability Major/Minor version number:    5.2
  Total amount of global memory:                 6143 MBytes (6441730048 bytes)
  (22) Multiprocessors, (128) CUDA Cores/MP:     2816 CUDA Cores
  GPU Max Clock rate:                            1076 MHz (1.08 GHz)
  Memory Clock rate:                             3505 Mhz
  Memory Bus Width:                              384-bit
  L2 Cache Size:                                 3145728 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(65536), 2D=(65536, 65536), 3D=(4096, 4096, 4096)
  Maximum Layered 1D Texture Size, (num) layers  1D=(16384), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(16384, 16384), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     Yes
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 3 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 8.0, CUDA Runtime Version = 7.5, NumDevs = 1, Device0 = GeForce GTX 980 Ti
Result = PASS
```

## Getting the other samples to run, getting CUDA to have graphics capabilities, soft symbolic linking to the existing libraries.

The flow or general procedure I ended up having to do was to use `locate` to find the relevant `*.so.*` or `*.h` file for the missing library or missing header, respectively, and then making soft symbolic links to them with the `ln -s` command.  I found that some of the samples have different configurations for in which directory the graphical libraries are (GL, GLU, X11, glut, etc.) than other samples in the samples included by NVIDIA.  


### Why do I see "nvcc: No such file or directory" when I try to build a CUDA application 

[NVIDIA CUDA Getting Started Guide for Linux - CUDA_Getting_Started_Linux.pdf](http://developer.download.nvidia.com/embedded/L4T/r23_Release_v2.0/CUDA/CUDA_Getting_Started_Linux.pdf?autho=1463194964_12deb625011712b74e3df117272a90ea&file=CUDA_Getting_Started_Linux.pdf)

"Your `LD_LIBRARY_PATH` environment variable is not set up correctly. Ensure that your `LD_LIBRARY_PATH` includes the lib and/or lib64 directory where you installed the Toolkit, usually `/usr/local/cuda-7.0/lib{,64}`:" (replace 7.0 with whatever version you're on; mine is 7.5)
`
$ export LD_LIBRARY_PATH=/usr/local/cuda-7.0/lib:$LD_LIBRARY_P
`

## Recovering from disastrous `dnf update` that adds a new kernel and trashes video output

<!-- [![`dnf update` after a long time, and having forgotten to **NOT** do this](https://scontent-lax3-1.cdninstagram.com/t50.2886-16/14936067_1878126812406187_9083488430947565568_n.mp4)] -->

<blockquote class="instagram-media" data-instgrm-captioned data-instgrm-version="7" style=" background:#FFF; border:0; border-radius:3px; box-shadow:0 0 1px 0 rgba(0,0,0,0.5),0 1px 10px 0 rgba(0,0,0,0.15); margin: 1px; max-width:658px; padding:0; width:99.375%; width:-webkit-calc(100% - 2px); width:calc(100% - 2px);"><div style="padding:8px;"> <div style=" background:#F8F8F8; line-height:0; margin-top:40px; padding:50.0% 0; text-align:center; width:100%;"> <div style=" background:url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAsCAMAAAApWqozAAAABGdBTUEAALGPC/xhBQAAAAFzUkdCAK7OHOkAAAAMUExURczMzPf399fX1+bm5mzY9AMAAADiSURBVDjLvZXbEsMgCES5/P8/t9FuRVCRmU73JWlzosgSIIZURCjo/ad+EQJJB4Hv8BFt+IDpQoCx1wjOSBFhh2XssxEIYn3ulI/6MNReE07UIWJEv8UEOWDS88LY97kqyTliJKKtuYBbruAyVh5wOHiXmpi5we58Ek028czwyuQdLKPG1Bkb4NnM+VeAnfHqn1k4+GPT6uGQcvu2h2OVuIf/gWUFyy8OWEpdyZSa3aVCqpVoVvzZZ2VTnn2wU8qzVjDDetO90GSy9mVLqtgYSy231MxrY6I2gGqjrTY0L8fxCxfCBbhWrsYYAAAAAElFTkSuQmCC); display:block; height:44px; margin:0 auto -44px; position:relative; top:-22px; width:44px;"></div></div> <p style=" margin:8px 0 0 0; padding:0 4px;"> <a href="https://www.instagram.com/p/BMNi4XWDIlM/" style=" color:#000; font-family:Arial,sans-serif; font-size:14px; font-style:normal; font-weight:normal; line-height:17px; text-decoration:none; word-wrap:break-word;" target="_blank">Ugh this is what happens when you allow dnf update to automatically install a new kernel that doesn&#39;t play well with your proprietary @nvidiageforce @nvidia drivers.  Time to troubleshoot. I better not lose the whole system.</a></p> <p style=" color:#c9c8cd; font-family:Arial,sans-serif; font-size:14px; line-height:17px; margin-bottom:0; margin-top:8px; overflow:hidden; padding:8px 0 7px; text-align:center; text-overflow:ellipsis; white-space:nowrap;">A video posted by Ernest Yeung (@ernestyalumni) on <time style=" font-family:Arial,sans-serif; font-size:14px; line-height:17px;" datetime="2016-10-31T03:07:54+00:00">Oct 30, 2016 at 8:07pm PDT</time></p></div></blockquote>
<script async defer src="//platform.instagram.com/en_US/embeds.js"></script>

<!-- <video width="100%" height="100%">
<source src="https://scontent-lax3-1.cdninstagram.com/t50.2886-16/14936067_1878126812406187_9083488430947565568_n.mp4" type="video/mp4"></video> -->

