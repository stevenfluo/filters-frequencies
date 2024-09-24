---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
title: Home
permalink: /
---

<style>
    /* Container for the entire row of images */
    .image-row {
      display: flex;
      justify-content: space-between; /* Space between images */
      flex-wrap: wrap; /* Ensure it wraps on smaller screens */
      margin: 20px;
    }

    /* Individual image container */
    .image-container {
      text-align: center; /* Center the caption */
      width: 30%; /* 22% width with margins gives space for 4 images */
      margin: 0 10px; /* Space between images */
    }

    /* Make images responsive */
    .image-container img {
      width: 100%;
      height: auto; /* Adjust image height based on width */
    }

    /* Style for captions */
    .caption {
      font-size: 12px;
      color: #555;
      margin-top: 5px;
    }
  </style>

#  CS 180: Project 2, Steven Luo

# Overview
In this project, we use filters to do cool things with images and convolutions, and explore hybrid image creation, image sharpening and multi-resolution blending with Gaussian and Laplacian stacks.  

# Part 1: Fun with Filters

## 1.1: Finite Difference Operator

How do we take the derivative (a continuous operation) of an image (a discrete space)? We can use the finite difference operator to estimate the image derivative, with an operator in each direction:

<p align="center">
    <img src="./math/fdo.png" alt="fdo" width="33%"/>
    <p style="text-align: center;"><i>Finite difference operators.</i></p>
</p>

By convolving the image with D_x and D_y, we effectively get the partial derivative showing changes in the vertical and horizontal directions. With this, we know the gradient, which we can use to calculate the edge strength (aka the gradient magnitude) by taking the square root of the sum of the squares of the partial derivatives:

<p align="center">
    <img src="./math/edge_strength.png" alt="edge strength" width="33%"/>
    <p style="text-align: center;"><i>Edge strength formula.</i></p>
</p>

Let's see the finite difference operator in action! Using an image of a cameraman, I convolve this with D_x and D_y to show the changes in each direction. For convolutions, I use `scipy.signal.convolve2d` with `mode = "same"`.

<p align="center">
    <img src="./img/cameraman.png" alt="cameraman" width="25%"/>
    <img src="./img/dx.png" alt="dx" width="25%"/>
    <img src="./img/dy.png" alt="dy" width="25%"/>
    <p style="text-align: center;"><i>Original cameraman photo, convolved with D_x, convolved with D_y.</i></p>
</p>

Next, I compute the gradient magnitude image, and turn this into an edge image by binarizing it with some threshold. After experimenting with a range of values, I found that using 0.25 as my threshold created the best-looking edge image. Setting a threshold allows us to binarize the gradient magnitude image while controlling how sensitive the image is to noise.

<p align="center">
    <img src="./img/web_camgradmag.jpg" alt="cameraman" width="20%"/>
    <img src="./img/web_camedges10.jpg" alt="dx" width="20%"/>
    <img src="./img/web_camedges25.jpg" alt="dy" width="20%"/>
    <img src="./img/web_camedges50.jpg" alt="dy" width="20%"/>
    <p style="text-align: center;"><i>Gradient magnitude image, edge image with threshold 0.10, threshold 0.25, threshold 0.50. The edge image with threshold = 0.25 best balanced noise removal and preserving image detail.</i></p>
</p>

## 1.2: Derivative of Gaussian (DoG) Filter

If we look at the above images, we can clearly see a lot of noise! We can use a low-pass filter to remove noise since it's a high-frequency component. We convolve our image with a Gaussian to smooth it and get rid of the noise, and along the way, I'll show how the associativity and communitivity of the convolution operation allow us to discover the same result using a different order of operations.

To create my Gaussian filter, I used `cv2.getGaussianKernel` with `kernel_size = 10` and `sigma = kernel_size / 6` (as discussed in lecture), and took the outer product of this with itself. Convolving our original image with our Gaussian filter gives us a smoother image and new partial derivatives. 

<p align="center">
    <img src="./img/cameraman.png" alt="cameraman" width="25%"/>
    <img src="./img/websmoothed.jpg" alt="img" width="25%"/>
    <p style="text-align: center;"><i>Original image, smoothed image.</i></p>
</p>

With some tuning, I found that the best threshold for this edge image was 0.055.

<!-- 
<p align="center">
    <img src="./img/smoothdx.png" alt="img" width="20%"/>
    <img src="./img/smoothdy.png" alt="img" width="20%"/>
    <img src="./img/webcam_gauss_grad_mag_im.jpg" alt="img" width="20%"/>
    <img src="./img/webcam_gauss_edges.jpg" alt="img" width="20%"/>
    <p style="text-align: center; font-size:75%"><i>Smoothed image convolved with D_x, convolved with D_y, gradient magnitude image, edge image with threshold = 0.055.</i></p>
</p> -->

<table>
  <tbody>
    <tr>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/smoothdx.png"><br />Smoothed image convolved with D_x.</td>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/smoothdy.png"><br />Smoothed image convolved with D_y.</td>
    </tr>
    <tr>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/webcam_gauss_grad_mag_im.jpg"><br />Gradient magnitude image.</td>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/webcam_gauss_edges.jpg"><br />Edge image with threshold=0.055</td>
    </tr>
  </tbody>
</table>

Some changes I noticed: the binarized edges are thicker and rounder than in that of the non-smoothed edge image; there's significantly less noise outside of the edge areas (like underneath the camera and in the grass); we lost some of the camera's fine detail.

Now, we use the derivative of Gaussian filters to do the same thing. We first convolve D_x or D_y with the Gaussian, then convolve the result with the image. 

<table>
  <tbody>
    <tr>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/dogdx.png"><br />Smoothed image convolved with derivative of Gaussian filter (D_x).</td>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/dogdy.png"><br />Smoothed image convolved with derivative of Gaussian filter (D_y).</td>
    </tr>
    <tr>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/webcam_dog_grad_mag_im.jpg"><br />Gradient magnitude image.</td>
      <td style="text-align: center;"><img width="200" alt="img" src="./img/webcam_dog_edges.jpg"><br />Edge image with threshold=0.055</td>
    </tr>
  </tbody>
</table>

The images are the same! In our first method, we are doing `(A * D_x) * G` where `A` is the original image, `D_x` is one of our finite difference operators, and `G` is the Gaussian filter. In our second method, we change the order by doing `A * (D_x * G)`, and our result shows that we get the same result. We would get the same result by doing `(G * D_x) * A` by commutivity and associativity of the convolution. [These](https://math.stackexchange.com/questions/2170534/proof-of-associativity-of-convolution) [posts](https://math.stackexchange.com/questions/4445/proving-commutativity-of-convolution-f-ast-gx-g-ast-fx) helped me when I was taking EECS126!

Here's what the derivative of Gaussian filters look like:

<p align="center">
    <img src="./img/dogfilterdx.png" alt="cameraman" width="25%"/>
    <img src="./img/dogfilterdy.png" alt="img" width="25%"/>
    <p style="text-align: center;"><i>DoG filter in the x- and y-directions.</i></p>
</p>

They're very small — only a 10x10 kernel!


# Part 2: Fun with Frequencies!

## 2.1: Image "Sharpening"

Image sharpening makes the edges in an image more noticeable by accentuating the high-frequency components. I implement this technique by subtracting the Gaussian filter-blurred image from the original, giving us the high frequencies of the image. The Gaussian filter works because it is a low pass filter: convolving the Gaussian with an image results in the low frequencies, so subtracting the low frequencies from the original image gives us the high frequencies. By scaling our isolated high frequencies and recombining it back into the original, we get a sharpened image. We express this mathematically as: 

`Sharpened Image = Original Image + α(Original Image - Low-Pass Filtered Image)`

α is a multiplier that controls the sharpness of the resulting image. Higher α's result in more heavily sharpened images.

<div class="image-row">
    <div class="image-container">
      <img src="./img/taj.jpg" alt="Image 1">
      <div class="caption">Original Taj</div>
    </div>
    <div class="image-container">
      <img src="./img/taj_sharp_2_10_1.6666666666666667.jpg" alt="Image 2">
      <div class="caption">Sharpened Taj: k=10, σ=10/6, α=2</div>
    </div>
    <div class="image-container">
      <img src="./img/taj_details_2_10_1.6666666666666667.jpg" alt="Image 3">
      <div class="caption">High frequencies</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/pocky.jpg" alt="Image 1">
      <div class="caption">Original Pocky the cat</div>
    </div>
    <div class="image-container">
      <img src="./img/pocky_sharp_6_100_2.jpg" alt="Image 2">
      <div class="caption">Sharpened Pocky: k=75, σ=15, α=4</div>
    </div>
    <div class="image-container">
      <img src="./img/pocky_details_6_100_2.jpg" alt="Image 3">
      <div class="caption">High frequencies</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/kittens.jpg" alt="Image 1">
      <div class="caption">Original kittens</div>
    </div>
    <div class="image-container">
      <img src="./img/kittens_sharp_2_100_25.jpg" alt="Image 2">
      <div class="caption">Sharpened kittens: k=100, σ=25, α=2</div>
    </div>
    <div class="image-container">
      <img src="./img/kittens_details_2_100_25.jpg" alt="Image 3">
      <div class="caption">High frequencies</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/goat.jpg" alt="Image 1">
      <div class="caption">Original 喜羊羊</div>
    </div>
    <div class="image-container">
      <img src="./img/goat_sharp_2_10_2.jpg" alt="Image 2">
      <div class="caption">Sharpened 喜羊羊: k=10, σ=2, α=2</div>
    </div>
    <div class="image-container">
      <img src="./img/goat_details_2_10_2.jpg" alt="Image 3">
      <div class="caption">High frequencies</div>
    </div>
</div>

Some of these images did not sharpen very well (specifically Pocky). I suspect it was because of my choice of Gaussian filter parameters and by the nature of the composition of the image - everything is already pretty fuzzy so isolating high frequencies would be harder.

For evaluation, I took a sharpened image, blurred it, and tried to sharpen it again. 

<!-- <div class="image-row">
    <div class="image-container">
      <img src="./img/house.jpg" alt="Image 1">
      <div class="caption">Original house</div>
    </div>
    <div class="image-container">
      <img src="./img/house_sharp_2_10_1.6666666666666667.jpg" alt="Image 2">
      <div class="caption">Sharpened house: k=10, σ=10/6, α=2</div>
    </div>
</div> -->

<p align="center">
    <img src="./img/house.jpg" alt="cameraman" width="40%"/>
    <img src="./img/house_sharp_2_10_1.6666666666666667.jpg" alt="img" width="40%"/>
    <p style="text-align: center;"><i>Original house, Sharpened house: k=10, σ=10/6, α=2</i></p>
</p>

<p align="center">
    <img src="./img/house_blurred_sharp_2_10_1.6666666666666667.jpg" alt="cameraman" width="40%"/>
    <img src="./img/house_blurred_details_2_10_1.6666666666666667.jpg" alt="img" width="40%"/>
    <p style="text-align: center;"><i>Blurred then sharpened house: k=10, σ=10/6, α=2; High frequencies</i></p>
</p>

<p align="center">
    <img src="./img/house_blurred_sharp_4_10_1.6666666666666667.jpg" alt="cameraman" width="40%"/>
    <img src="./img/house_blurred_sharp_16_10_1.6666666666666667.jpg" alt="img" width="40%"/>
    <p style="text-align: center;"><i>α=4; α=16</i></p>
</p>

No matter how we adjust alpha, the resulting image is not as good as the original. In this case, we lose the "warmness" and "roundness" of parts of the house (especially on the cobblestone, by the waves, and the candle glow), and amping up the high frequencies does not replace what the blur took away. I also noticed that when I sharpened the image using the high frequencies of the original, I saw a bigger change in the image than when sharpening with the details of the blurred image. This shows how we cannot rely on our sharpening technique to undo an image blur if we do not have the details from the original image.

<!-- <div class="image-row">
    <div class="image-container">
      <img src="" alt="Image 1">
      <div class="caption">Blurred then sharpened house: k=10, σ=10/6, α=2</div>
    </div>
    <div class="image-container">
      <img src="" alt="Image 2">
      <div class="caption">Blurred then sharpened house: k=10, σ=10/6, α=4</div>
    </div>
    <div class="image-container">
      <img src="" alt="Image 3">
      <div class="caption">Blurred then sharpened house: k=10, σ=10/6, α=8</div>
    </div>
</div> -->

## 2.2: Hybrid Images

To create a hybrid image, we combine a low pass filtered image (only low frequencies remain) with a high pass filtered image (only high frequencies remain). These images are static, appear to change as the viewing distance changes. Though high frequencies tend to dominate perception, at a distance only the low frequency part of the signal can be seen. By blending the high frequency portion of one image with the low-frequency portion of another, you get a hybrid image that leads to different interpretations at different distances.

I started with the provided images of `Derek.jpg` and `Nutmeg.jpg`, using the provided alignment code then low-passing Derek and high-passing Nutmeg (by subtracting the low-frequencies out of the original image). I used a kernel size of 50 for both Gaussian filters, but settled on `sigma = 12` for the low pass and `sigma = 10` for the high pass after some tuning. By averaging the results, I made this hybric image:

<!-- <p align="center">
    <img src="./img/derekcat.jpg" alt="cameraman" width="30%"/>
    <p style="text-align: center;"><i>Derekmeg?</i></p>
</p> -->

<div class="image-row">
    <div class="image-container">
      <img src="./img/derekcat_im1.jpg" alt="Image 1">
      <div class="caption">Aligned Derek</div>
    </div>
    <div class="image-container">
      <img src="./img/derekcat.jpg" alt="Image 2">
      <div class="caption">Derekmeg?</div>
    </div>
    <div class="image-container">
      <img src="./img/derekcat_im1low.jpg" alt="Image 2">
      <div class="caption">Low-passed Derek</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/derekcat_im2.jpg" alt="Image 1">
      <div class="caption">Aligned Nutmeg</div>
    </div>
    <div class="image-container">
      <img src="./img/derekcat_im2high.jpg" alt="Image 2">
      <div class="caption">High-passed Nutmeg</div>
    </div>
</div>

Some other photos I made: 

Articuno + Pidgey: `kernel_size = 30` `low_sigma = 2`, `high_sigma = 8` 

<div class="image-row">
    <div class="image-container">
      <img src="./img/articuno.jpg" alt="Image 1">
      <div class="caption">Articuno</div>
    </div>
    <div class="image-container">
      <img src="./img/pidgey.jpg" alt="Image 2">
      <div class="caption">Pidgey</div>
    </div>
    <div class="image-container">
      <img src="./img/p_hybrid.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

This one didn't turn out that well: I could have tuned the Gaussian parameters and cutoff frequency better, but I think the bigger issue is that the proportions of the images don't match up and the contrasting colors are visually too contrasting, making it hard for them to "blend" into a hybrid image.

Happy + Sad: `kernel_size = 30` `low_sigma = 15`, `high_sigma = 5` 

<div class="image-row">
    <div class="image-container">
      <img src="./img/sad.jpg" alt="Image 1">
      <div class="caption">Sad face</div>
    </div>
    <div class="image-container">
      <img src="./img/happy.jpg" alt="Image 2">
      <div class="caption">Happy face</div>
    </div>
    <div class="image-container">
      <img src="./img/smilesad.png" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

This one also didn't turn out that well: the features on each image are so distinct it's not possible to line one set of features up without ruining another (i.e. I aligned based on the circle, at the cost of misaligned eyes). Tweaking Gaussian parameters on the high-passed image might help it blend in a bit better, but I'm not convinced that it'll solve the structural issues of very discrete shapes in the image.

Charli XCX + Lorde: `kernel_size = 30` `low_sigma = 2`, `high_sigma = 4` 

<div class="image-row">
    <div class="image-container">
      <img src="./img/charli2.jpg" alt="Image 1">
      <div class="caption">Charli XCX</div>
    </div>
    <div class="image-container">
      <img src="./img/lorde.jpg" alt="Image 2">
      <div class="caption">Lorde</div>
    </div>
    <div class="image-container">
      <img src="./img/lordhybrid.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

I really like how this one turned out — thanks to my friend for the suggestion!

Magikarp + Feebas: `kernel_size = 30` `low_sigma = 2`, `high_sigma = 2` 

<div class="image-row">
    <div class="image-container">
      <img src="./img/feebas.jpg" alt="Image 1">
      <div class="caption">Feebas</div>
    </div>
    <div class="image-container">
      <img src="./img/magikarp.jpg" alt="Image 2">
      <div class="caption">Magikarp</div>
    </div>
    <div class="image-container">
      <img src="./img/fishhybrid.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

Sather Gate, then and now: `kernel_size = 30` `low_sigma = 3`, `high_sigma = 10` 

<div class="image-row">
    <div class="image-container">
      <img src="./img/satherold.jpg" alt="Image 1">
      <div class="caption">Sather Gate (old)</div>
    </div>
    <div class="image-container">
      <img src="./img/sathernew1.jpg" alt="Image 2">
      <div class="caption">Sather Gate (recent)</div>
    </div>
    <div class="image-container">
      <img src="./img/satherhybrid.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

This one was really interesting, because you can see how students now walk on the same path through the gate that the Free Speech Movement protestors demonstrated by.

Favorite Result: Donald Swift? `kernel_size = 30` `low_sigma = 6`, `high_sigma = 6` 

Creepy or cool... you be the judge! Frequency analysis is below (log magnitudes of the Fourier transform are shown).

<div class="image-row">
    <div class="image-container">
      <img src="./img/trump2.jpg" alt="Image 1">
      <div class="caption">Donald Trump</div>
    </div>
    <div class="image-container">
      <img src="./img/swift.jpg" alt="Image 2">
      <div class="caption">Taylor Swift</div>
    </div>
    <div class="image-container">
      <img src="./img/donaldswift.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/tayloralign.jpg" alt="Image 1">
      <div class="caption">Aligned Taylor</div>
    </div>
    <div class="image-container">
      <img src="./img/donaldalign.jpg" alt="Image 2">
      <div class="caption">Aligned Donald</div>
    </div>
    <div class="image-container">
      <img src="./img/donaldswift.jpg" alt="Image 2">
      <div class="caption">Hybrid</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/fft_2.png" alt="Image 1">
      <div class="caption">Aligned Taylor FFT</div>
    </div>
    <div class="image-container">
      <img src="./img/fft_im1.png" alt="Image 2">
      <div class="caption">Aligned Donald FFT</div>
    </div>
    <div class="image-container">
      <img src="./img/fft_hybrid.png" alt="Image 2">
      <div class="caption">Hybrid FFT</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/taylorhigh.jpg" alt="Image 1">
      <div class="caption">High-passed Taylor</div>
    </div>
    <div class="image-container">
      <img src="./img/donaldlow.jpg" alt="Image 2">
      <div class="caption">Low-passed Donald</div>
    </div>
</div>

<div class="image-row">
    <div class="image-container">
      <img src="./img/fft_high2.png" alt="Image 1">
      <div class="caption">High-passed Taylor FFT</div>
    </div>
    <div class="image-container">
      <img src="./img/fft_low1.png" alt="Image 2">
      <div class="caption">Low-passed Donald FFT</div>
    </div>
</div>

Bells & Whistles: I used color to enhance the effect of the hybrid image. I tried using color for both, then only the high or the low frequency image, and I found that using color for either both components or the low frequency component only worked best. When you filter an image to only its highest frequencies, you eliminate most of the visible color spectrum, which is why I suspect using color on the high frequency image doesn't make much of a difference. For future improvement, I would crop the images better.

<div class="image-row">
    <div class="image-container">
      <img src="./img/taylorcolorhybrid.jpg" alt="Image 1">
      <div class="caption">High frequency color only</div>
    </div>
    <div class="image-container">
      <img src="./img/donaldcolorhybrid.jpg" alt="Image 2">
      <div class="caption">Low frequency color only</div>
    </div>
    <div class="image-container">
      <img src="./img/dtgrayhybrid.jpg" alt="Image 2">
      <div class="caption">Grayscale</div>
    </div>
</div>

# Multi-resolution Blending and the Oraple Journey

## 2.3: Gaussian and Laplacian Stacks

I implemented Gaussian and Laplacian stacks. Layers in the Gaussian stack are created by convolving the previous layer with a Gaussian filter, initializing the stack with the original image. Laplacian stack layers are the difference between successive layers in the Gaussian stack. I was careful to avoid off-by-one errors in my implementation, and especially so when using the Gaussian stack created from the mask. I found the project-referenced paper [Burt and Adelson (1983)](https://persci.mit.edu/pub_pdfs/spline83.pdf) and lecture slides visualizing the stacks to be helpful while building my implementation. These stacks allow us to blend images without the "ghosting" we see as a result of alpha blending.

For the oraple, my parameters were: 8 layers, `kernel size = 30`, `sigma = 5`. 

Gaussian and Laplacian Stacks of Apple:
<p align="center">
    <img src="./img/apples.png" alt="cameraman" width="100%"/>
</p>

Gaussian and Laplacian Stacks of Orange:
<p align="center">
    <img src="./img/oranges.png" alt="cameraman" width="100%"/>
</p>

Recreation of Figure 3.42: the first three rows show the high, medium, and low-frequency parts of the Laplacian stack (taken from levels 0, 2, and 4 respectively). The left and middle columns show the original apple and orange images convolved with the corresponding layer from the mask Gaussian stack, while the right column shows the combined image. The last row is the result of flattening the stack on the left half, right half, and combined images respectively. 

<p align="center">
    <img src="./img/fig42.png" alt="cameraman" width="50%"/>
</p>

A more intuitive way to think about combining the images in the stack is as follows: we squash the columns rightwards to get the image in the last column, and we can squash the rows downwards to get the image in the last row. Note that by summing the Laplacian stack of an image, we can recover the original!

## 2.4: Multiresolution Blending (a.k.a. the oraple!)

My blended images can be found below. I use a Gaussian filter with `kernel_size = 30` and `sigma = 5` to create the Gaussian stack for the input images, but in some cases I instead use a larger filter on the masks for a stronger blur effect (specifically on the oraple, where I use `kernel_size = 50` and `sigma = 25` to create the mask Gaussian stack only). 

Oraple:
<div class="image-row">
    <div class="image-container">
      <img src="./img/orange.jpeg" alt="Image 1">
      <div class="caption">Orange</div>
    </div>
    <div class="image-container">
      <img src="./img/apple.jpeg" alt="Image 2">
      <div class="caption">Apple</div>
    </div>
    <div class="image-container">
      <img src="./img/oraple_mask.jpg" alt="Image 2" border="1">
      <div class="caption">Mask</div>
    </div>
</div>


<p align="center">
    <img src="./img/oraple.png" alt="Oraple!" width="30%"/>
    <p style="text-align: center;"><i>Oraple!</i></p>
</p>

Harris/Trump:
<div class="image-row">
    <div class="image-container">
      <img src="./img/harris.jpg" alt="Image 1">
      <div class="caption">Kamala Harris</div>
    </div>
    <div class="image-container">
      <img src="./img/trump3.jpg" alt="Image 2">
      <div class="caption">Donald Trump</div>
    </div>
    <div class="image-container">
      <img src="./img/harris_trump_mask.jpg" alt="Image 2" border="1">
      <div class="caption">Mask</div>
    </div>
</div>


<p align="center">
    <img src="./img/kamalatrump.png" alt="Oraple!" width="30%"/>
    <img src="./img/donaldharris.png" alt="Oraple!" width="30%"/>
    <p style="text-align: center;"><i>Kamala Trump, Donald Harris</i></p>
</p>

Charli the Apple?
<div class="image-row">
    <div class="image-container">
      <img src="./img/charli.jpg" alt="Image 1">
      <div class="caption">Charli XCX</div>
    </div>
    <div class="image-container">
      <img src="./img/apple.jpg" alt="Image 2">
      <div class="caption">Apple</div>
    </div>
    <div class="image-container">
      <img src="./img/apple_mask.jpg" alt="Image 2" border="1">
      <div class="caption">Mask</div>
    </div>
</div>


<p align="center">
    <img src="./img/charpple.png" alt="Oraple!" width="30%"/>
    <p style="text-align: center;"><i>Charli Apple (is this brat?)</i></p>
</p>

Does SF City Hall really look like the U.S. Congress?
<div class="image-row">
    <div class="image-container">
      <img src="./img/sf.jpg" alt="Image 1">
      <div class="caption">San Francisco City Hall</div>
    </div>
    <div class="image-container">
      <img src="./img/congress.jpg" alt="Image 2">
      <div class="caption">Congress</div>
    </div>
    <div class="image-container">
      <img src="./img/sf_congress_mask.jpg" alt="Image 2" border="1">
      <div class="caption">Mask</div>
    </div>
</div>


<p align="center">
    <img src="./img/sfcongress.png" alt="Oraple!" width="30%"/>
    <p style="text-align: center;"><i>California, D.C.?</i></p>
</p>

The proportions and shapes of the two buildings and images were a bit different, so this was hard to blend well and I don't count this as a total success. I do see the resemblance, though!

Campanile @ Stanford:
<div class="image-row">
    <div class="image-container">
      <img src="./img/campanile.jpg" alt="Image 1">
      <div class="caption">Campanile (Berkeley)</div>
    </div>
    <div class="image-container">
      <img src="./img/stanford.jpg" alt="Image 2">
      <div class="caption">Hoover Tower (Stanford)</div>
    </div>
    <div class="image-container">
      <img src="./img/tower_mask.jpg" alt="Image 2" border="1">
      <div class="caption">Mask</div>
    </div>
</div>


<p align="center">
    <img src="./img/towertower.png" alt="Oraple!" width="30%"/>
    <p style="text-align: center;"><i>Stanford looks a lot cooler here...</i></p>
</p>

The Laplacian stack generated during this process is below. The first column is of the Hoover Tower image Laplacians after convolving with the mask, the second column is of the Campanile, the third column is the layer-combined image, and the fourth column is the corresponding mask from the mask's Gaussian stack. The last row consists of stack-combined images (except the fourth column where the mask is just the original mask).

<p align="center">
    <img src="./img/grid.png" alt="Oraple!" width="90%" />
</p>

I tried to find an image of Stanford's Hoover Tower at a similar angle on a clearer day, but somehow couldn't anywhere online! The backgrounds would have blended much more nicely.


Bells & Whistles: The color images are shown above, and are significantly more appealing than the grayscale versions.

A note on setting filter sizes: for this project, I took Prof. Efros' advice from lecture that a general rule of thumb for setting Gaussian filter sizes is that the filter half-width should be about `3 * σ` (also in the "Convolution and Image Derivatives" lecture).

## Reflection
I had a lot of fun picking images and blending them together. The most important thing I learned from this project is that the frequencies in an image really affect our perception and understanding of what we see, and that image editing is a lot harder than it looks. Also, having the tools to superimpose images onto other images gives a lot of power (and responsibility) to the user, and makes me think more about how higher-tech parallels (like deepfakes) interact with society and how we perceive our visual surroundings. I got to make a lot of silly images to share, though!

<!-- <p align="center">
    <img src="" alt="img" width="20%"/>
    <img src="" alt="img" width="20%"/>
    <img src="" alt="img" width="20%"/>
    <img src="" alt="img" width="20%"/>
    <p style="text-align: center;"><i>Caption.</i></p>
</p> -->