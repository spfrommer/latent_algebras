---
# layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<link rel="stylesheet" type="text/css" href="https://fonts.googleapis.com/css?family=Raleway">


<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$','$'], ['\\(','\\)']],
    processEscapes: true
  }
});
</script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>


<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Transport of Algebraic Structure</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Raleway";
    font-weight: 300;
    font-size:18px;
    line-height: 4.0;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
    color: #333332;
  }
  
  h1 {
    font-weight:300;
    font-size: 35px;
  }
  h2 {
    font-weight:300;
    font-size: 25px;
    margin-top: 10px;
    margin-bottom: 10px;
  }
  h3 {
    font-weight:250;
    font-size: 25px;
  }

p {
    font-size: 20px;
    line-height: 1.4;
}
A:link {
    COLOR: #5432c2
}
A:visited {
    COLOR: #5432c2
}

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;  
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}

hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table 
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<!--<link rel="shortcut icon" type="image/x-icon" href="favicon.ico">-->
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center>
<h1><strong>
<!-- Ditto <img width="50" style='display:inline;' src="./src/ditto.png"/> <br> -->
Transport of Algebraic Structure to Latent Embeddings
</strong></h1></center>
<center><h2>
    &nbsp;&nbsp;&nbsp;<a href="https://sam.pfrommer.us/">Samuel Pfrommer</a>&nbsp;&nbsp; 
    <a href="https://brendon-anderson.github.io/">Brendon Anderson</a>&nbsp;&nbsp;
    <a href="https://people.eecs.berkeley.edu/~sojoudi/">Somayeh Sojoudi</a>&nbsp;&nbsp;
   </h2>
    <center><h2>
        University of California, Berkeley&nbsp;&nbsp;&nbsp;
    </h2></center>
    <center><h2>
        ICML 2024 (Spotlight) &nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/abs/2405.16763">Paper</a> | <a href="https://github.com/spfrommer/latent_algebras">Code</a> </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Machine learning often aims to produce latent embeddings of inputs which lie in a larger, abstract mathematical space. For example, in the field of 3D modeling, subsets of Euclidean space can be embedded as vectors using implicit neural representations. Such subsets also have a natural algebraic structure including operations (e.g., union) and corresponding laws (e.g., associativity). How can we learn to “union” two sets using only their latent embeddings while respecting associativity? We propose a general procedure for parameterizing latent space operations that are provably consistent with the laws on the input space. This is achieved by learning a bijection from the latent space to a carefully designed <em>mirrored algebra</em> which is constructed on Euclidean space in accordance with desired laws. We evaluate these <em>structural transport nets</em> for a range of mirrored algebras against baselines that operate directly on the latent space. Our experiments provide strong evidence that respecting the underlying algebraic structure of the input space is key for learning accurate and self-consistent operations. 
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h2 align="center">Algebraic structure in embedding spaces</h2>

<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Many learned embeddings derive from mathematical objects with <em>algebraic structure</em>.

Consider the space of subsets of $\mathbb{R}^n$, which can be endowed with algebraic operations consisting of <em>operations</em> (e.g., $\cap$, $\cup$) and <em>laws</em> (e.g., $A \cup B = B \cup A$).

Any particular set in this space can be captured as the sublevel set of a learned network, i.e., an Implicit Neural Representation (INR); we can extract latent embeddings of sets via a autoencoder-style architecture on INR weights.

A similar idea applies to learned functions, probability distributions, and textual embeddings.
</p></td></tr></table>
</p>
  </div>


<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
    <img src="figs/intro.png" alt="Algebraic structure overview" style="width:700px;">
    </td>
  </tr>
  </tbody>
</table>


<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
These operations are often foundational for a range of downstream tasks where we may only have access to latent embeddings. This leads us to ask a key question:
</p></td></tr></table>
</p>
  </div>

<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="center" width="20%">
<em>Can we learn operations on the latent space which provably respect the algebraic laws of the underlying space?</em>
</p></td></tr></table>
</p>
  </div>

<hr>

<h2 align="center">Transport of structure</h2>


<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
A precise formulation of our method 
</p></td></tr></table>
</p>
  </div>

<hr>

<h2 align="center">Method: Intervention-based Reweighting Scheme</h2>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
    </td>
  </tr>
  </tbody>
</table>

<hr>

<h2 align="center">Experiment Results</h2>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">Our system ensures safe and reliable execution through human-robot teaming. We evaluated the autonomous policy performance of our human-in-the-loop framework on 4 tasks. As the autonomous policy improves over long-term deployment, the amount of human workload decreases.
</p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>

</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">We conduct 3 rounds of robot deployments and policy updates. Here we present Round 1 and Round 3 results of Ours and baseline IWR. We show how for Ours policy performance improve over rounds, and how Ours outperforms IWR baseline. </p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>

</tbody>
</table>

<h2 align="center">Gear Insertion (Real)</h2>

<h3 align="center">Ours, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>


<h3 align="center">IWR, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">Ours, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h2 align="center">Coffee Pod packing (Real)</h2>

<h3 align="center">Ours, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 1</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">Ours, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<h3 align="center">IWR, Round 3</h3>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle">
  </td>
  </tr>
  </tbody>
</table>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={arXiv preprint arXiv:2202.08227},
   year={2022}
}
</code></pre>
</left></td></tr></table>

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->
