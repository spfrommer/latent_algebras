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
MathJax.Hub.Register.StartupHook("End Jax",function () {
  var BROWSER = MathJax.Hub.Browser;
  var jax = "CommonHTML";
  if (BROWSER.isMSIE && BROWSER.hasMathPlayer) jax = "NativeMML";
  return MathJax.Hub.setRenderer(jax);
});
</script>
<script async="async" src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS-MML_HTMLorMML"></script>
<!--<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>-->


<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Transport of Algebraic Structure</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<!--<meta property="og:title" content="TITLE">-->

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: sans-serif;
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

<style>
  .site-header {
    display: none;
  }
</style>

<style>
  .site-footer {
    display: none;
  }
</style>

<!--<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script-->
<!--src="./src/b5m.js" id="b5mmain"-->
<!--type="text/javascript"></script><script type="text/javascript"-->
<!--async=""-->
<!--src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>-->


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
    <a href="https://brendon-anderson.github.io/">Brendon G. Anderson</a>&nbsp;&nbsp;
    <a href="https://people.eecs.berkeley.edu/~sojoudi/">Somayeh Sojoudi</a>&nbsp;&nbsp;
   </h2>
    <center><h2>
        University of California, Berkeley&nbsp;&nbsp;&nbsp;
    </h2></center>
    <center><h2>
        ICML 2024 (Spotlight) &nbsp;&nbsp;&nbsp; 		
    </h2></center>
	<center><h2><a href="https://arxiv.org/abs/2405.16763"><img height="20" style='display:inline;' src="./icons/pdf.png"/> Paper</a> | <a href="https://github.com/spfrommer/latent_algebras"><img height="20" style='display:inline;' src="./icons/github.png"/> Code</a> </h2></center>


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

<h2 align="center">Algebraic structure and embedding spaces</h2>

<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Learned embeddings often represent mathematical objects with <em>algebraic structure</em>.

Consider the space of subsets of $\mathbb{R}^n$, which can be endowed with <em>operations</em> (e.g., $\cap$, $\cup$) and <em>laws</em> (e.g., $A \cup B = B \cup A$).

Implicit Neural Representations (INRs) capture subsets as the sublevel sets of learned networks; we can then extract latent embeddings $z$ of sets via an autoencoder-style architecture on INR weights $\phi$.

Specifically, we assume that we are provided with encoder/decoder maps $E$ and $D$ connecting latent embeddings in $L=\mathbb{R}^l$ to elements of the power set $\mathcal{P}(\mathbb{R}^n)$, where the latter <em>source space</em> carries algebraic structure.
</p></td></tr></table>
</p>
  </div>

<img src="figs/autoencoder.png" alt="INR autoencoder" style="width:700px;">


<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
The source algebra operations (e.g., $\cup$ and $\cap$) are often foundational for downstream tasks where we may only have access to latent embeddings. We thus intuitively want to learn latent-space operations which yield the correct results in set space. Informally, for the union operation this would mean that

    $$ \phantom{\mathrm{(informal)}} \quad \quad A \cup^{\mathcal{S}} B \approx D(E(A) \cup^{\mathcal{L}} E(B)), \quad \quad \mathrm{(informal)}$$

where $\cup^\mathcal{S}: \mathcal{P}(\mathbb{R}^n) \times \mathcal{P}(\mathbb{R}^n) \to \mathcal{P}(\mathbb{R}^n)$ is the standard set union and $\cup^{\mathcal{L}}: L \times L \to L$ is a learned latent-space analog. We could imagine directly parameterizing maps $\cup^{\mathcal{L}}: L \times L \to L$ and $\cap^{\mathcal{L}}: L \times L \to L$ as MLPs:
</p></td></tr></table>
</p>
  </div>

<img src="figs/operations.png" alt="Learned operations" style="width:700px;">

<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
But such a naive parameterization would not capture the symmetries provided by the laws of the source algebra; $D(E(A) \cup^{\mathcal{L}} E(B))$ and $D(E(B) \cup^{\mathcal{L}} E(A))$ could yield completely different sets! This leads us to ask a key question:
</p></td></tr></table>
</p>
  </div>

<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="center" width="20%">
<em>Can we learn operations on the latent space which provably respect the algebraic laws of the source algebra?</em>
</p></td></tr></table>
</p>
  </div>


<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
While we focus our work on sets for exposition, we note that this connection between latent embeddings and source-space algebraic structure is quite general. A similar idea applies to learned (continuous) functions, probability distributions, and textual embeddings.
</p></td></tr></table>
</p>
  </div>

<img src="figs/intro.png" alt="Algebraic structure overview" style="width:700px;">


<hr>

<h2 align="center">Transport of structure</h2>


<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
A precise formulation of our method relies on machinery from universal algebra. We provide an informal explanation here and refer to the <a href="https://arxiv.org/abs/2405.16763">full paper</a> for additional details.
</p></td></tr></table>
</p>
  </div>



<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
Our key idea is to parameterize a <em>learned bijection</em> $\varphi: L \to M$, where the <em>mirrored space</em> $M = \mathbb{R}^l$ is of the same dimensionality as $L$. We endow $M$ with one operation for each operation on the source algebra, attempting to ensure that these operations satisfy all required laws. For instance, we can define mirrored-space operations $\cup^{\mathcal{M}}: M \times M \to M, \cup^{\mathcal{M}}: M \times M \to M$ which form a <em>Riesz space</em>:

$$ \begin{aligned} a \cup^{\mathcal{M}} b &= \mathrm{max}(a, b), \\ a \cap^{\mathcal{M}} b &= \mathrm{min}(a, b). \end{aligned} $$

This choice of operations satisfies all the same laws as $\cup^{\mathcal{S}}$ and $\cap^{\mathcal{S}}$ do for the space of sets: commutativity, associativity, absorption, and distributivity. We can then transfer the operations $f \in \{\cup, \cap\}$ to the latent space via $\varphi$:

$$
	f^{\mathcal{L}}(z_1,\dots,z_n) := \varphi^{-1}\big(f^{\mathcal{M}}(\varphi(z_1),\dots,\varphi(z_n))\big).
$$

Crucially, since $\varphi$ is a learned bijection, a good choice of operations on $M$ yields operations on $L$ which <em>provably satisfy</em> the desired source algebra laws. Details on how we train $\varphi$ and choose mirrored-space operations are deferred to the full paper.

<img src="figs/transport.png" alt="Algebraic structure overview" style="width:700px;">

<hr>

<h2 align="center">Experimental results</h2>


<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">
Our problem setting concerns learning $\cap$ and $\cup$ over a dataset of synthetic sets in $\mathbb{R}^2$. We evaluate a variety of mirrored space operation combinations and two directly parameterized MLP references. 
</p>
</td>
</tr>
</tbody>
</table>


<img src="figs/results.png" alt="Results" style="width:700px;">

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">
The left-hand figure above plots the performance of learned operations against the number of satisfied source algebra laws. Each scatter point represents one combination of learned operations, with the solid line capturing the mean. The increasing trend confirm our primary hypothesis: learned latent-space operations achieve higher performance when constructed to satisfy source algeba laws.
</p>
</td>
</tr>
</tbody>
</table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr><td>
  <p align="justify" width="20%">
  The right-hand figure examines the <em>self-consistency</em> of learned operations; do they yield similar results for equivalent terms? For instance, does the prediction for $A \cup B$ match $B \cup A$? The Riesz space transported algebra is perfectly self-consistent as it satisfies all source algebra laws. However, all other learned operations degrade as more random laws are applied.
</p>
</td>
</tr>
</tbody>
</table>

<hr>

<h2 align="center">Reference</h2>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@inproceedings{pfrommer2024transport,
   title={Transport of Algebraic Structure to Latent Embeddings},
   author={Pfrommer, Samuel and Anderson, Brendon G and Sojoudi, Somayeh},
   booktitle={International Conference on Machine Learning},
   year={2024}
}
</code></pre>
</left></td></tr></table>

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<!--<div style="display:none">-->
<!-- Global site tag (gtag.js) - Google Analytics -->
<!--<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>-->
<!--<script>-->
  <!--window.dataLayer = window.dataLayer || [];-->
  <!--function gtag(){dataLayer.push(arguments);}-->
  <!--gtag('js', new Date());-->

  <!--gtag('config', 'G-PPXN40YS69');-->
<!--</script>-->

<script>
var links = document.links;
for (var i = 0; i < links.length; i++) {
     links[i].target = "_blank";
}
</script>
<!-- </center></div></body></div> -->
