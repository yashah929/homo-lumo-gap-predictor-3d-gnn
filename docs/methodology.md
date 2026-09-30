# Methodology

## Problem definition and information boundary

For each molecule, the objective is to predict the direct QM9 HOMO–LUMO gap \(y=\Delta E_{\mathrm{HOMO-LUMO}}\) in eV. Inputs consist of molecular composition, covalent connectivity, local chemical categories, and equilibrium Cartesian geometry. The implementation excludes partial charges, orbital energies, total or atomization energies, and any descriptor derived from a quantum-mechanical target calculation. Although QM9 provides HOMO and LUMO energies, neither is an input or a separately learned target.

PyTorch Geometric stores QM9 properties in a fixed vector. The code asserts that index 4 is named `gap` before extracting it. The original hydrogen-explicit SDF is used to obtain RDKit chemical attributes in the same atom order as PyG coordinates; atomic-number sequences are compared and processing fails on a mismatch.

## Graph definition

Let a molecule contain atoms \(V=\{1,\ldots,N\}\), atomic positions \(\mathbf r_i\in\mathbb R^3\), and chemical bonds \(B\). The directed graph is

\[
G=(V,E),\qquad E=\{(j,i):i,j\in V,\ j\ne i\}.
\]

Consequently, \(|E|=N(N-1)\). Self-edges are unnecessary because the GRU hidden state supplies a gated self-information path. No spatial cutoff removes an edge.

The distance on each directed pair is

\[
d_{ji}=\lVert\mathbf r_j-\mathbf r_i\rVert_2.
\]

Distances are invariant to global translation, rotation, and reflection. Raw coordinate components are never supplied to a learned layer.

## Atomic features

Atomic number \(Z_i\) uses a learned embedding. Seven additional categories use separate learned embeddings:

1. explicit graph degree (0–6 plus unknown);
2. formal charge (−3 through +3 plus unknown);
3. hybridization (`UNSPECIFIED`, `S`, `SP`, `SP2`, `SP3`, `SP3D`, `SP3D2`, `OTHER`, unknown);
4. aromaticity (binary);
5. total valence (0–8 plus unknown);
6. ring membership (binary);
7. chiral tag (unspecified, tetrahedral clockwise, tetrahedral counterclockwise, other).

Atomic mass in daltons is the sole continuous node attribute and is divided by 100 as a fixed numerical scale. It is not fitted from any data split. All embeddings and scaled mass are concatenated and passed through a two-layer SiLU projection to produce \(h_i^{(0)}\in\mathbb R^d\). Explicit hydrogen atoms are retained and featurized identically.

## Geometric and chemical edge representation

Fifty RBF centers are uniformly spaced over \([a,b]=[0,10]\) Å by default:

\[
\mu_k=a+k\delta,\qquad \delta=\frac{b-a}{49},\qquad k=0,\ldots,49.
\]

The deterministic width parameter is

\[
\gamma=\delta^{-2},
\]

and the expansion is

\[
\phi_k(d_{ji})=\exp[-\gamma(d_{ji}-\mu_k)^2].
\]

At one neighboring center's displacement a basis has value \(e^{-1}\), producing smooth overlap. Values outside the interval are evaluated without clipping; preprocessing reports the maximum distance and, by default, raises if it exceeds \(b+0.1\) Å.

The chemical vector \(c_{ji}\in\mathbb R^7\) contains a bonded indicator; single, double, triple, and aromatic indicators; conjugation; and bond-ring membership. If \(\{i,j\}\notin B\), then \(c_{ji}=0\). Geometry and bond information form one representation on one topology:

\[
z_{ji}=[\phi_0(d_{ji}),\ldots,\phi_{49}(d_{ji}),c_{ji}].
\]

## Continuous-filter message passing

For layer \(t\), an edge network with SiLU activations and configurable dropout (0.10 by default) creates a hidden-dimensional filter:

\[
g_{ji}^{(t)}=\operatorname{MLP}^{(t)}_{\mathrm{edge}}(z_{ji})\in\mathbb R^d.
\]

A linear map produces the sender value \(v_j^{(t)}=W^{(t)}h_j^{(t)}\). The pairwise message and permutation-invariant incoming aggregation are

\[
m_{ji}^{(t)}=g_{ji}^{(t)}\odot v_j^{(t)},\qquad
m_i^{(t)}=\sum_{j\ne i}m_{ji}^{(t)}.
\]

Every layer has independent edge-network, sender-projection, GRU, and LayerNorm parameters. Depth \(T\in\{3,4,5,6\}\) is selected by cross-validation.

## Recurrent update

The aggregate is the GRU input and the previous atom representation is its recurrent hidden state:

\[
\widetilde h_i^{(t+1)}=\operatorname{GRUCell}^{(t)}(m_i^{(t)},h_i^{(t)}),
\qquad
h_i^{(t+1)}=\operatorname{LayerNorm}^{(t)}(\widetilde h_i^{(t+1)}).
\]

No external residual is added because the GRU update gate already controls retention of the previous state. Dropout is not applied to the recurrent hidden transition. Batch normalization is not used.

## Molecular readout and prediction

PyG Set2Set applies three rounds of recurrent, content-based attention to the unordered set \(\{h_i^{(T)}\}\). If \(q_t\) is an LSTM query, attention and readout can be summarized as

\[
\alpha_{i,t}=\operatorname{softmax}_i(h_i^{(T)\top}q_t),\qquad
r_t=\sum_i\alpha_{i,t}h_i^{(T)},\qquad
q_t^*=[q_t,r_t].
\]

The final molecular vector \(q_3^*\) has dimension \(2d\). The regression head is

\[
2d\xrightarrow{\mathrm{Linear,SiLU,Dropout}}d
\xrightarrow{\mathrm{Linear,SiLU,Dropout}}d/2
\xrightarrow{\mathrm{Linear}}1.
\]

Both dropout probabilities are 0.10 by default. The scalar is a standardized gap prediction.

## Target transformation and optimization

For a training index set \(S\), only its targets determine

\[
\mu_S=\frac{1}{|S|}\sum_{n\in S}y_n,
\qquad
\sigma_S=\sqrt{\frac{1}{|S|}\sum_{n\in S}(y_n-\mu_S)^2},
\qquad
y_n'=\frac{y_n-\mu_S}{\sigma_S}.
\]

The population standard deviation is used. The training objective is standardized MSE,

\[
\mathcal L=\frac1{|\mathcal B|}\sum_{n\in\mathcal B}(\widehat y_n'-y_n')^2.
\]

Predictions are inverted as \(\widehat y_n=\sigma_S\widehat y_n'+\mu_S\) before all reported metrics:

\[
\mathrm{MAE}=\frac1M\sum_n|y_n-\widehat y_n|,
\quad
\mathrm{RMSE}=\sqrt{\frac1M\sum_n(y_n-\widehat y_n)^2},
\]

\[
R^2=1-\frac{\sum_n(y_n-\widehat y_n)^2}{\sum_n(y_n-\bar y)^2}.
\]

MAE and RMSE are in eV. Adam uses \(\beta_1=0.9\), \(\beta_2=0.999\), \(\epsilon=10^{-8}\), with cosine annealing.

## Split, selection, and final evaluation

A seed-42 permutation partitions all retained molecules once: the first \(\lfloor0.8N\rfloor\) form the development set and the remainder form the locked final test set. A second deterministic permutation of development indices is divided with `numpy.array_split` into four folds. These validation sets are mutually disjoint and exhaustive over development.

For configuration \(c\), model selection uses

\[
\overline{\mathrm{MAE}}_c=\frac14\sum_{k=1}^4\mathrm{MAE}_{c,k}.
\]

Fold training statistics, early stopping (validation MAE, patience 30), and best checkpoints are independent. The 36-member grid crosses depth \(\{3,4,5,6\}\), width \(\{64,128,256\}\), and learning rate \(\{10^{-4},3\times10^{-4},10^{-3}\}\). Selection uses mean MAE, its sample standard deviation, smaller width, and shallower depth in that order.

For the selected configuration, \(E_{\mathrm{final}}\) is the nearest integer to the median of its four best epochs, with half values rounded upward. A new seed-4242 initialization is trained on all development samples for exactly this duration using development-only normalization. The separate final command then evaluates the unchanged checkpoint once on locked test indices.
