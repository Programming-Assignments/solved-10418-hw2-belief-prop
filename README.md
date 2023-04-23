Download Link: https://assignmentchef.com/product/solved-10418-hw2-belief-prop
<br>
<h1>1      Written Questions</h1>

Answer the following questions in the template provided. Then upload your solutions to Gradescope. You may use L<sup>A</sup>T<sub>E</sub>X or print the template and hand-write your answers then scan it in. Failure to use the template may result in a penalty. There are 66 points and 16 questions.

<h2>1.1      Conditional Independencies</h2>

<ol>

 <li>Consider the Bayesian Network described in Figure 1</li>

</ol>

Figure 1.1: Bayesian Network Structure

Based on this network structure, answer the following questions:

(a) (1 point) Write down the equation for the joint probability distribution <em>P</em>(<em>A,B,C,D,E,F,G</em>)

True

False

<ul>

 <li>(1 point) Is <em>A </em>⊥ <em>F </em>| <em>B</em>?</li>

</ul>

True

False

<ul>

 <li>(1 point) Is <em>A </em>⊥ <em>G </em>| <em>B</em>?</li>

</ul>

True

False

<ul>

 <li>(1 point) Which nodes are present in the Markov blanket of <em>B</em>?</li>

 <li>(1 point) Which nodes are present in the Markov blanket of <em>D</em>?</li>

</ul>

<ol start="2">

 <li>Now consider an undirected graphical model with the same set of nodes and edges as the bayesian network from figure 2. This model structure looks as follows:</li>

</ol>

Figure 1.2: Undirected Graphical Model

For this model structure, answer the following questions:

<ul>

 <li>(1 point) Is <em>C </em>⊥ <em>D </em>| <em>E</em>?</li>

</ul>

True

False

<ul>

 <li>(1 point) Is <em>A </em>⊥ <em>F </em>| <em>B</em>?</li>

</ul>

True

False

<ul>

 <li>(1 point) Is <em>A </em>⊥ <em>G </em>| <em>B</em>?</li>

</ul>

True

False

<ul>

 <li>(1 point) Which nodes are present in the Markov blanket of <em>B</em>?</li>

 <li>(1 point) Which nodes are present in the Markov blanket of <em>D</em>?</li>

</ul>

<ol start="3">

 <li>Let us now compare both models (1 and 1.2).

  <ul>

   <li>(1 point) Do both models (1 and 1.2) have the same set of conditional independencies?</li>

  </ul></li>

</ol>

Yes

No

<ul>

 <li>(2 points) If you answered yes to the above question, list out all the conditional independencies. If you answered no, provide an example of a graph which does have the same set of conditional independencies for both directed and undirected variants.</li>

 <li>(2 points) For the directed bayesian network, we decomposed the joint probability distribution into a product of conditional probability distributions associated with each node. However, we did not do so for the undirected model. Is it possible to write joint probability as a product of factors <em>without </em>performing marginalization (i.e. no summations) for a general undirected graphical model? Explain your answer.</li>

</ul>

<h2>1.2      Variable Elimination</h2>

<ol>

 <li>In class, we looked at an example of variable elimination on an arbitrary graph. Let us now apply variable elimination to a familiar directed graphical model: Hidden Markov Model. A Hidden Markov Model consists of two sets of variables: <em>X<sub>i </sub></em>(observations) and <em>Y<sub>i </sub></em>(states). States are unobserved latent variables which satisfy the Markov property i.e. each state only depends on the state which immediately precedes it. Each state generates an observation. The complete structure of the model (for a sequence of length 5) looks as follows:</li>

</ol>

Figure 1.3: Hidden Markov Model

<ul>

 <li>(2 points) Draw the corresponding factor graph for this model.</li>

</ul>

<table width="570">

 <tbody>

  <tr>

   <td width="570">Latex users: If you want to use tikz to draw the factor graph, here is a sample code snippet for a tiny factor graph:tikz[square/.style={regular polygon,regular polygon sides=4}] {
ode[latent] (A) {A};
ode[latent,right=1.5 cm of A] (B) {B};
ode[square, draw=black, right=0.5 cm of A] (ab) {};edge [-] {A} {ab};edge [-] {B} {ab};}This snippet generates the following graph:</td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(2 points) For this model, write down the joint probability distribution as a product of conditional probability distributions.</li>

 <li>(4 points) Suppose we wish to compute the probability <em>P</em>(<em>Y</em><sub>5 </sub>| <em>X</em><sub>1</sub><em>…X</em><sub>5</sub>), which requires us to marginalize over <em>Y</em><sub>1</sub><em>…Y</em><sub>4</sub>. Assume that we are eliminating variables in the order <em>Y</em><sub>1 </sub>− <em>Y</em><sub>2 </sub>− <em>Y</em><sub>3 </sub>− <em>Y</em><sub>4</sub>. Write down equations for the factors which will be computed at each step of the elimination process.</li>

</ul>

<table width="498">

 <tbody>

  <tr>

   <td width="142">Variable Eliminated</td>

   <td width="357">Factor Computed</td>

  </tr>

  <tr>

   <td width="142"><em>Y</em><sub>1</sub></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>Y</em><a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>Y</em><sub>3</sub></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>Y</em><sub>4</sub></td>

   <td width="357"> </td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(1 point) Is it possible to pick a better elimination order for this model?</li>

</ul>

Yes

No

<ul>

 <li>(0 points) Do you observe any similarities between the factors computed during variable elimination and the standard forward-backward algorithm for HMMs?</li>

 <li>(2 points) Draw a factor graph for this model, with each factor corresponding to a maximal clique in the graph</li>

 <li>(4 points) For the variable elimination order <em>A</em>−<em>G</em>−<em>B </em>−<em>D</em>−<em>E </em>−<em>F </em>−<em>C</em>, draw the intermediate factor graph at each step</li>

</ul>




<table width="498">

 <tbody>

  <tr>

   <td width="142">Variable Eliminated</td>

   <td width="357">Intermediate Factor Graph</td>

  </tr>

  <tr>

   <td width="142"><em>A</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>G</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>B</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>D</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>E</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>F</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>C</em></td>

   <td width="357"> </td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(4 points) For the variable elimination order <em>C </em>−<em>B </em>−<em>E </em>−<em>A</em>−<em>D</em>−<em>F </em>−<em>G</em>, draw the intermediate factor graph at each step</li>

</ul>




<table width="498">

 <tbody>

  <tr>

   <td width="142">Variable Eliminated</td>

   <td width="357">Intermediate Factor Graph</td>

  </tr>

  <tr>

   <td width="142"><em>C</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>B</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>E</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>A</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>D</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>F</em></td>

   <td width="357"> </td>

  </tr>

  <tr>

   <td width="142"><em>G</em></td>

   <td width="357"> </td>

  </tr>

 </tbody>

</table>

<ul>

 <li>(1 point) Which of the above elimination orders is better and why?</li>

 <li>(1 point) Based on your observations, can you think of a way to estimate which elimination order is better without going through the complete process?</li>

</ul>

<h2>1.3     Message Passing</h2>

Figure 1.5

Consider the factor graph in Figure 1.5. On paper, carry out a run of belief propagation by sending messages first from the leaves <em>ψ<sub>A</sub>,ψ<sub>B</sub>,ψ<sub>C </sub></em>to the root <em>ψ<sub>E</sub></em>, and then from the root back to the leaves. Then answer the questions below. Assume all messages are un-normalized.

<ol>

 <li>(1 point) Numerical answer: What is the message from <em>A </em>to <em>ψ<sub>EA</sub></em>?</li>

 <li>(1 point) Numerical answer: What is the message from <em>ψ<sub>DB </sub></em>to <em>B</em>?</li>

 <li>(1 point) Numerical answer: What is the belief at variable <em>A</em>?</li>

 <li>(1 point) Numerical answer: What is the belief at variable <em>B</em>?</li>

</ol>

<h2>1.4      Empirical Questions</h2>

The following questions should be completed after you work through the programming portion of this assignment (Section 2).

<ol>

 <li>(1 point) Select one: If you feed the inputs shown in Figure 5 into your belief propagation module implemented in PyTorch do you get the same answers that you worked out on paper? <em>(Hint: The correct answer is “Yes”.)</em></li>

</ol>

Yes

No

<ol start="2">

 <li>(10 points) Record your model’s performance on the test set and train set in terms of Cross Entropy (CE) , accuracy (AC) and leaf accuracy (LAC). <em>Note: Round each numerical value to two significant figures.</em></li>

</ol>

<table width="245">

 <tbody>

  <tr>

   <td width="98">Schedule</td>

   <td width="92">Baseline</td>

   <td width="54">CRF</td>

  </tr>

  <tr>

   <td width="98">Training CE</td>

   <td width="92"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="98">Training AC</td>

   <td width="92"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="98">Training LAC</td>

   <td width="92"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="98">Test AC</td>

   <td width="92"> </td>

   <td width="54"> </td>

  </tr>

  <tr>

   <td width="98">Test LAC</td>

   <td width="92"> </td>

   <td width="54"> </td>

  </tr>

 </tbody>

</table>

<ol start="3">

 <li>(10 points) Plot training and testing cross entropy curves for : <em>Baseline</em>, <em>CRF Model</em>. Let the <em>x</em>-axis ranges over 3 epochs. <em>Note: Your plot must be machine generated.</em></li>

 <li>(1 point) Multiple Choice: Did you correctly submit your code to Autolab?</li>

</ol>

Yes

No

<ol start="2">

 <li>(1 point) Numerical answer: How many hours did you spend on this assignment?.</li>

</ol>

<h1>2      Programming [30 pts]</h1>

Your goal in this assignment is to implement a CRF belief propagation algorithm for constituency parsing. Given the structure of the tree, you will implement a model to tag the nodes with appropriate POS tag.

Your solution for this section must be implemented in PyTorch using the data files we have provided to you. This restriction is because we will be grading your code by hand to check your understanding as well as your model’s performance.

<h2>2.1      Task Background</h2>

Constituency parsing aims to extract a parse tree from a sentence that represents its syntactic structure according to a phrase structure grammar. Non-terminals in the tree are types of phrases, the terminals are the words in the sentence, and the edges are unlabeled. Throughout this assignment, we use nodes to refer to the set of all non terminals.

Figure 2.1: A parse tree with eight nodes – four leaves and four intermediate nodes.

Assuming we know the structure of this tree, our goal is to successfully predict the appropriate tag for all its nodes. We will train our model with a Cross Entropy Loss, based on the beliefs of each node after message passing. We define the accuracy of the model as the average accuracy over all examples where each example consists of a tree structure with <em>n </em>nodes. Accuracy for a single tree is defined as

number of correctly predicted nodes

Acc =

total number of nodes in the tree

Note that this accuracy is computed across all nodes in the graph. We also define leaf accuracy as

number of correctly predicted leaf nodes

Leaf Acc = total number of leaves in the tree

<h2>2.2     Data</h2>

We have provided a preprocessed version of Penn Tree Bank with 49,208 examples. Each line consists of one tree. You are to split the data using a 70/30 ratio for train and test.

We have provided starter code to read each line into an NLTK tree data structure, and a custom tree structure, which you can modify.

Given Input: An input sequence and the associated skeleton of its constituency parse tree.

Goal Output: The labels of the non terminals in the parse tree. 2.3 Baseline

Figure 2.2: Our baseline model using a unidirectional LSTM.

For this section, you must implement a working baseline LSTM model.

Our baseline model uses the hidden state of an LSTM layer to predict the output tag. For the leaf nodes, this computation is trivial. For intermediate nodes, we use a linear layer on the concatenation of the hidden state of the LSTM outputs of the left and the right child to compute distributions over tags.

Your implementation must have a single layer unidirectional LSTM with a hidden dimension of 128. The embedding size should be 256. Set your optimizer to be Adam with a learning rate of 0.0001. Due to the complexity associated with building the tree and computing its potentials, you can use a batch size of 1. Your program should be able to run on a laptop without GPUs due to the simplicity of our model.

<h2>2.4      Adding the CRF layer</h2>

For this section, you must implement functions to compute the unary potentials for each node and binary potential for each edge in the tree.

The CRF layer consists of computing unary node potentials and binary edge potentials derived from the LSTM hidden state. You need to compute a unary potential for every node in the graph and an edge potential for every edge. Note here that the dotted line between the non terminals and the terminals do not count as edges.

Recall that these potentials are the scores associated for each of the possible tags, and have to be positive. By simply using a linear layer on the output of the LSTMs, however, we may get negative scores. To account for these negative values, we assume that the output of the linear layer is the logarithm of potentials. All further computation is carried out in log space. This gives us the added advantage of numerical stability[

2.6].

<h2>2.5     Message passing implementation</h2>

For this section, you must implement a function for belief propagation.

The sum-product message passing algorithm is defined as follows: While there is a node <em>x<sub>i </sub></em>ready to transmit to <em>x<sub>j</sub></em>, send the message

<em>m<sub>i</sub></em>→<em>j </em>= <sup>X</sup><em>φ</em>(<em>x<sub>i</sub></em>)<em>φ</em>(<em>x<sub>i</sub>,x<sub>j</sub></em>) <sup>Y </sup><em>m<sub>l</sub></em>→<em>i</em>(<em>x<sub>i</sub></em>)

<em>x</em><em>i                                                   </em><em>l</em>∈<em>N</em>(<em>i</em>)<em>/j</em>

Here, <em>N</em>(<em>i</em>)<em>/j </em>refers to the set of nodes that are neighbors of i, excluding j. After we have computed all messages, we may answer any marginal query over <em>x<sub>i </sub></em>in constant time using the equation.

<em>p</em>(<em>x<sub>i</sub></em>) ∝ <em>φ</em>(<em>x<sub>i</sub></em>) <sup>Y </sup><em>m<sub>l</sub></em>→<em><sub>i</sub></em>(<em>x<sub>i</sub></em>)

<em>l</em>∈<em>N</em>(<em>i</em>)

We will be implementing the asynchronous version of the algorithm in this assignment. This consists of two sets of messages, an upward pass from the leaves to the root, and a downward pass from the root node to the leaf nodes.

<h2>2.6       The LogSumExp trick</h2>

The LogSumExp trick is a common trick in machine learning to deal with problems of numerical stability. To illustrate the problem, consider the contrived example for our features <em>x<sub>i</sub></em>: [1000, 1000, 1000]. Feeding this sequence into the softmax function should yield a probability distribution of [1/3, 1/3, 1/3] and the log of 1/3 is a reasonable negative number. However, calculating one of the terms of the summation in python yields the following output:

To deal with the underflow (and similar overflow), we can compute all messages and potentials in the log space. Multiplication operations on messages would then turn to addition of log messages using

<em>e</em><em>a</em><em>.e</em><em>b </em>= <em>e</em><em>a</em>+<em>b</em>

and log(<em>ab</em>) = log(<em>a</em>) + log(<em>b</em>)

Hint: Pytorch has a logsumexp function which can be applied to any dimension of a tensor.

<h2>2.7     Test time decoding</h2>

During test time decoding, predict the tag with highest marginal probability for each variable. DO NOT run the belief propagation here.

<h2>2.8      Autolab Submission [30 pts]</h2>

You must submit a .tar file named beliefprop.tar containing beliefprop.py, which contains all of your code.

You can create that file by running:

tar -cvf beliefprop.tar beliefprop.py

from the directory containing your code.

Some additional tips: DO NOT compress your files; you are just creating a tarball. Do not use tar -czvf. DO NOT put the above files in a folder and then tar the folder. Autolab is case sensitive, so observe that all your files should be named in lowercase. You must submit this file to the corresponding homework link on Autolab.

Your code will not be autograded on Autolab. Instead, we will grade your code by hand; that is, we will read through your code in order to grade it. As such, please carefully identify major sections of the code via comments.

<a href="#_ftnref1" name="_ftn1">[1]</a> . In class, we saw how using variable elimination is more efficient than naively computing the joint probability. In this problem, we will further study how the order in which variable elimination is carried out affects the efficiency of this method. Consider the following undirected graphical model:

Figure 1.4: Initial graph for variable elimination