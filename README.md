# Rotation-Equivariant Neural Jacobian Fields for Generalizable 3D Mesh Deformations


### SIGGRAPH Asia 2026


**[Lukas Uzolas](https://lukas.uzolas.com/)<sup>1</sup>, 
[Ruben Wiersma](https://rubenwiersma.nl/)<sup>2,3</sup>,
[Olga Sorkine-Hornung](https://igl.ethz.ch/people/sorkine/)<sup>2</sup>,
[Petr Kellnhofer](https://kellnhofer.xyz/)<sup>3</sup>**
<br>
TU Delft<sup>1</sup>, ETH Zurich<sup>2</sup>, Adobe Research<sup>3</sup>
<br>


Neural Jacobian Fields are state-of-the-art neural representations for deformations on 3D meshes.
A fundamental limitation of both the original formulation and follow-up works is that the learned Jacobians are not equivariant to ambient-space rotations of the source mesh. 
This hinders generalization to shapes that are misaligned or greatly different in shape.
Moreover, our analysis shows that this rotation dependence stems from the formulation itself and cannot be remedied with rotation-invariant features alone.
Instead, we propose a rotation-equivariant formulation of Neural Jacobian Fields, Re-NJF, representing Jacobians in local frames of triangle faces aligned with their normals.
The frames are learned end-to-end in the gradient space of semantic surface features distilled from 2D image foundation models. 
Robust to rotations and large geometry changes, these features support learning tangent frames and Jacobians that generalize across transformations in SO(3). 
Moreover, our localized tangent frame definition also facilitates generalization to non-rigid transformations of the input and substantial shape changes (e.g., dog to human).
We evaluate our method on human and animal pose deformation datasets to simultaneously demonstrate competitive performance in the well-explored aligned data regime, as well as superior performance in tasks requiring generalization across ambient space rotations and input shape variations.
