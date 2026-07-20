The empirical sample-based optimization problem is $$\begin{equation}
\label{eq:ipw-objective-full}
\max_{\theta}\;
\hat V_{\mathrm{IPW}}(\theta)
:=
\frac{1}{N}\sum_{i=1}^N
\sum_{t\in\mathcal T}
\pi_{\theta,t}(X_i)\,
\frac{\mathbf{1}\{T_i=t\}Y_i}{\hat e_t(X_i)}.
\end{equation}$$

Equivalently, this can be written as $$\begin{equation}
\label{eq:ipw-objective-short}
\max_{\theta}\;
\hat V_{\mathrm{IPW}}(\theta)
=
\frac{1}{N}\sum_{i=1}^N
\pi_{\theta,T_i}(X_i)\,
\frac{Y_i}{\hat e_{T_i}(X_i)}.
\end{equation}$$

The policy is defined by $$\begin{equation}
\label{eq:softmax-policy}
\pi_{\theta,t}(x)
=
\frac{
\exp\!\left(\bigl(m_{t,\theta}(x)-\mu_{t,\theta}\bigr)/\tau\right)
}{
\sum_{s\in\mathcal T}
\exp\!\left(\bigl(m_{s,\theta}(x)-\mu_{s,\theta}\bigr)/\tau\right)
},
\qquad t\in\mathcal T.
\end{equation}$$

For each fixed $\theta$, define the sample-based shadow-price vector
$\mu_\theta$ by $$\begin{equation}
\label{eq:mu-theta-sample}
\mu_{\theta}
\in
\arg\min_{\mu \in \mathbb{R}^{|\mathcal{T}|+1}_{+}}
\left\{
\frac{1}{N}\sum_{i=1}^N
\sum_{t\in\mathcal T}
\frac{
\exp\!\left((m_{t,\theta}(X_i)-\mu_t)/\tau\right)
}{
\sum_{s\in\mathcal T}\exp\!\left((m_{s,\theta}(X_i)-\mu_s)/\tau\right)
}
\bigl(m_{t,\theta}(X_i)-\mu_t\bigr)
+
\sum_{t\in\mathcal T}\mu_t b_t
\right\}.
\end{equation}$$

To optimize
[\[eq:ipw-objective-short\]](#eq:ipw-objective-short){reference-type="eqref"
reference="eq:ipw-objective-short"} with gradient-based methods, we
compute the derivative of the objective with respect to $\theta$. Since
the policy $\pi_{\theta,t}(x)$ depends on $\mu_\theta$, and $\mu_\theta$
is itself defined implicitly by the optimization problem
[\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
reference="eq:mu-theta-sample"}, this is a bilevel optimization problem.

For convenience, define $$\begin{equation}
\label{eq:w-i-def}
w_i := \frac{Y_i}{\hat e_{T_i}(X_i)},
\qquad i=1,\dots,N,
\end{equation}$$ and let $$\begin{equation}
\label{eq:objective-J}
J(\theta)
:=
\hat V_{\mathrm{IPW}}(\theta)
=
\frac{1}{N}\sum_{i=1}^N
\pi_{\theta,T_i}(X_i)\, w_i.
\end{equation}$$

Next, define the score $$\begin{equation}
\label{eq:z-def}
z_{\theta,t}(x)
:=
\frac{m_{t,\theta}(x)-\mu_{\theta,t}}{\tau},
\qquad t\in\mathcal T,
\end{equation}$$ so that the policy in
[\[eq:softmax-policy\]](#eq:softmax-policy){reference-type="eqref"
reference="eq:softmax-policy"} can be written as $$\begin{equation}
\label{eq:softmax-z}
\pi_{\theta,t}(x)
=
\frac{\exp(z_{\theta,t}(x))}
{\sum_{s\in\mathcal T}\exp(z_{\theta,s}(x))}.
\end{equation}$$

By the chain rule, $$\begin{equation}
\label{eq:grad-J-chain}
\nabla_\theta J(\theta)
=
\frac{1}{N}\sum_{i=1}^N
w_i \, \nabla_\theta \pi_{\theta,T_i}(X_i).
\end{equation}$$

For the softmax function, the derivative with respect to the logits is
$$\begin{equation}
\label{eq:softmax-deriv-logit}
\frac{\partial \pi_{\theta,t}(x)}{\partial z_{\theta,r}(x)}
=
\pi_{\theta,t}(x)\bigl(\mathbf{1}\{t=r\}-\pi_{\theta,r}(x)\bigr),
\qquad r,t\in\mathcal T.
\end{equation}$$ Therefore, $$\begin{equation}
\label{eq:grad-pi-chain}
\nabla_\theta \pi_{\theta,t}(x)
=
\sum_{r\in\mathcal T}
\frac{\partial \pi_{\theta,t}(x)}{\partial z_{\theta,r}(x)}
\,\nabla_\theta z_{\theta,r}(x).
\end{equation}$$

Using [\[eq:z-def\]](#eq:z-def){reference-type="eqref"
reference="eq:z-def"}, we have $$\begin{equation}
\label{eq:grad-z}
\nabla_\theta z_{\theta,r}(x)
=
\frac{1}{\tau}
\left(
\nabla_\theta m_{r,\theta}(x)
-
\nabla_\theta \mu_{\theta,r}
\right).
\end{equation}$$ Substituting
[\[eq:softmax-deriv-logit\]](#eq:softmax-deriv-logit){reference-type="eqref"
reference="eq:softmax-deriv-logit"} and
[\[eq:grad-z\]](#eq:grad-z){reference-type="eqref"
reference="eq:grad-z"} into
[\[eq:grad-pi-chain\]](#eq:grad-pi-chain){reference-type="eqref"
reference="eq:grad-pi-chain"} yields $$\begin{equation}
\label{eq:grad-pi-final}
\nabla_\theta \pi_{\theta,t}(x)
=
\frac{1}{\tau}
\sum_{r\in\mathcal T}
\pi_{\theta,t}(x)\bigl(\mathbf{1}\{t=r\}-\pi_{\theta,r}(x)\bigr)
\left(
\nabla_\theta m_{r,\theta}(x)
-
\nabla_\theta \mu_{\theta,r}
\right).
\end{equation}$$

It remains to compute $\nabla_\theta \mu_\theta$. To this end, define
the inner sample-based objective from
[\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
reference="eq:mu-theta-sample"} as $$\begin{equation}
\label{eq:g-theta-mu}
g(\theta,\mu)
:=
\frac{1}{N}\sum_{i=1}^N
\sum_{t\in\mathcal T}
\frac{
\exp\!\left((m_{t,\theta}(X_i)-\mu_t)/\tau\right)
}{
\sum_{s\in\mathcal T}\exp\!\left((m_{s,\theta}(X_i)-\mu_s)/\tau\right)
}
\bigl(m_{t,\theta}(X_i)-\mu_t\bigr)
+
\sum_{t\in\mathcal T}\mu_t b_t.
\end{equation}$$

Assume for now that $\mu_\theta$ is an interior minimizer of
[\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
reference="eq:mu-theta-sample"} and that $g(\theta,\mu)$ is twice
continuously differentiable. Then $\mu_\theta$ satisfies the first-order
condition $$\begin{equation}
\label{eq:foc-mu}
\nabla_\mu g(\theta,\mu_\theta)=0.
\end{equation}$$ Differentiating
[\[eq:foc-mu\]](#eq:foc-mu){reference-type="eqref"
reference="eq:foc-mu"} with respect to $\theta$ gives $$\begin{equation}
\label{eq:implicit-diff-step}
\nabla^2_{\mu\mu} g(\theta,\mu_\theta)\,
\frac{d\mu_\theta}{d\theta}
+
\nabla^2_{\mu\theta} g(\theta,\mu_\theta)
=
0.
\end{equation}$$ Hence, $$\begin{equation}
\label{eq:dmu-dtheta}
\frac{d\mu_\theta}{d\theta}
=
-
\Bigl(\nabla^2_{\mu\mu} g(\theta,\mu_\theta)\Bigr)^{-1}
\nabla^2_{\mu\theta} g(\theta,\mu_\theta).
\end{equation}$$

Substituting [\[eq:dmu-dtheta\]](#eq:dmu-dtheta){reference-type="eqref"
reference="eq:dmu-dtheta"} into
[\[eq:grad-z\]](#eq:grad-z){reference-type="eqref"
reference="eq:grad-z"}, then into
[\[eq:grad-pi-final\]](#eq:grad-pi-final){reference-type="eqref"
reference="eq:grad-pi-final"}, and finally into
[\[eq:grad-J-chain\]](#eq:grad-J-chain){reference-type="eqref"
reference="eq:grad-J-chain"}, yields the full gradient of the empirical
IPW objective: $$\begin{equation}
\label{eq:full-gradient}
\nabla_\theta J(\theta)
=
\frac{1}{N}\sum_{i=1}^N
w_i\,
\frac{1}{\tau}
\sum_{r\in\mathcal T}
\pi_{\theta,T_i}(X_i)
\bigl(\mathbf{1}\{T_i=r\}-\pi_{\theta,r}(X_i)\bigr)
\left(
\nabla_\theta m_{r,\theta}(X_i)
-
\nabla_\theta \mu_{\theta,r}
\right),
\end{equation}$$ where $\nabla_\theta \mu_{\theta,r}$ is obtained from
the implicit differentiation formula
[\[eq:dmu-dtheta\]](#eq:dmu-dtheta){reference-type="eqref"
reference="eq:dmu-dtheta"}.

# Differentiating the Inner Optimization Problem

In order to optimize the empirical IPW objective in
[\[eq:ipw-objective-short\]](#eq:ipw-objective-short){reference-type="eqref"
reference="eq:ipw-objective-short"}, we require the derivative of the
policy $\pi_{\theta,t}(x)$ with respect to $\theta$. Since the policy
depends on the shadow-price vector $\mu_\theta$, and $\mu_\theta$ is
defined implicitly as the solution to the inner optimization problem
[\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
reference="eq:mu-theta-sample"}, this requires differentiating through
an optimization problem.

We adopt the perspective of differentiable optimization layers and
compute the derivative of $\mu_\theta$ via implicit differentiation of
its optimality conditions.

## Notation

Let $K := |\mathcal T|$. For each observation $i\in\{1,\dots,N\}$,
define $$\begin{equation}
\label{eq:ai-def}
a_i(\theta,\mu)
:=
\bigl(a_{i,t}(\theta,\mu)\bigr)_{t\in\mathcal T}
\in \mathbb R^K,
\qquad
a_{i,t}(\theta,\mu)
=
m_{t,\theta}(X_i)-\mu_t.
\end{equation}$$

Define the softmax vector $$\begin{equation}
\label{eq:pii-def}
\pi_i(\theta,\mu)
=
\operatorname{softmax}\!\left(\frac{a_i(\theta,\mu)}{\tau}\right),
\end{equation}$$ and $$\begin{equation}
\label{eq:si-def}
s_i(\theta,\mu)
=
\pi_i(\theta,\mu)^\top a_i(\theta,\mu).
\end{equation}$$

Then the inner objective can be written as $$\begin{equation}
\label{eq:g-def}
g(\theta,\mu)
=
\frac{1}{N}\sum_{i=1}^N s_i(\theta,\mu)
+
\mu^\top b,
\end{equation}$$ where $b = (b_t)_{t\in\mathcal T}\in\mathbb R^K$.

## Optimality Condition

Assuming that $\mu_\theta$ is an interior minimizer of
[\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
reference="eq:mu-theta-sample"}, it satisfies the first-order optimality
condition $$\begin{equation}
\label{eq:foc}
\nabla_\mu g(\theta,\mu_\theta)=0.
\end{equation}$$

Define $$\begin{equation}
\label{eq:F-def}
F(\mu,\theta)
:=
\nabla_\mu g(\theta,\mu).
\end{equation}$$ Then $\mu_\theta$ is characterized implicitly by
$$\begin{equation}
\label{eq:implicit-system}
F(\mu_\theta,\theta)=0.
\end{equation}$$

## Implicit Differentiation

Under standard regularity conditions (smoothness and invertibility of
the Jacobian $\nabla_\mu F$), the Implicit Function Theorem implies that
$\mu_\theta$ is locally differentiable and satisfies $$\begin{equation}
\label{eq:implicit-linear-system}
\nabla^2_{\mu\mu} g(\theta,\mu_\theta)\,\frac{d\mu_\theta}{d\theta}
=
-
\nabla^2_{\mu\theta} g(\theta,\mu_\theta).
\end{equation}$$

Rather than explicitly forming the inverse Hessian, we compute
$\frac{d\mu_\theta}{d\theta}$ by solving the linear system
[\[eq:implicit-linear-system\]](#eq:implicit-linear-system){reference-type="eqref"
reference="eq:implicit-linear-system"}. This is the standard approach in
differentiable optimization layers.

## Derivative Structure

To evaluate
[\[eq:implicit-linear-system\]](#eq:implicit-linear-system){reference-type="eqref"
reference="eq:implicit-linear-system"}, we express derivatives of $g$
via the intermediate quantities $$\begin{equation}
\label{eq:psi-def}
\psi_i(\theta,\mu)
:=
\nabla_{a_i} s_i(\theta,\mu),
\qquad
H_i(\theta,\mu)
:=
\nabla^2_{a_i a_i} s_i(\theta,\mu).
\end{equation}$$

Using the chain rule and the relation $a_i(\theta,\mu)=m_i(\theta)-\mu$,
we obtain $$\begin{equation}
\label{eq:grad-mu}
\nabla_\mu g(\theta,\mu)
=
b-\frac{1}{N}\sum_{i=1}^N \psi_i(\theta,\mu),
\end{equation}$$ and $$\begin{equation}
\label{eq:grad-theta}
\nabla_\theta g(\theta,\mu)
=
\frac{1}{N}\sum_{i=1}^N
J_i(\theta)^\top \psi_i(\theta,\mu),
\end{equation}$$ where $$\begin{equation}
\label{eq:Ji}
J_i(\theta)
=
\frac{\partial m_i(\theta)}{\partial \theta}.
\end{equation}$$

Similarly, the second derivatives take the form $$\begin{equation}
\label{eq:hess-mumu}
\nabla^2_{\mu\mu} g(\theta,\mu)
=
\frac{1}{N}\sum_{i=1}^N H_i(\theta,\mu),
\end{equation}$$ and $$\begin{equation}
\label{eq:hess-mutheta}
\nabla^2_{\mu\theta} g(\theta,\mu)
=
-\frac{1}{N}\sum_{i=1}^N
H_i(\theta,\mu)\,J_i(\theta).
\end{equation}$$

Substituting [\[eq:hess-mumu\]](#eq:hess-mumu){reference-type="eqref"
reference="eq:hess-mumu"} and
[\[eq:hess-mutheta\]](#eq:hess-mutheta){reference-type="eqref"
reference="eq:hess-mutheta"} into
[\[eq:implicit-linear-system\]](#eq:implicit-linear-system){reference-type="eqref"
reference="eq:implicit-linear-system"}, we obtain $$\begin{equation}
\label{eq:final-linear-system}
\left(
\frac{1}{N}\sum_{i=1}^N H_i(\theta,\mu_\theta)
\right)
\frac{d\mu_\theta}{d\theta}
=
\frac{1}{N}\sum_{i=1}^N
H_i(\theta,\mu_\theta)\,J_i(\theta).
\end{equation}$$

## Remarks on Computation

The computation of gradients proceeds in two stages:

- **Forward pass:** solve the inner optimization problem
  [\[eq:mu-theta-sample\]](#eq:mu-theta-sample){reference-type="eqref"
  reference="eq:mu-theta-sample"} to obtain $\mu_\theta$.

- **Backward pass:** compute $\frac{d\mu_\theta}{d\theta}$ by solving
  the linear system
  [\[eq:final-linear-system\]](#eq:final-linear-system){reference-type="eqref"
  reference="eq:final-linear-system"}, and then apply the chain rule to
  obtain $\nabla_\theta J(\theta)$.

Importantly, the Hessian $\nabla^2_{\mu\mu} g(\theta,\mu_\theta)$ is
never inverted explicitly; instead, the linear system is solved using
standard numerical linear algebra routines. Since the number of
treatments $K$ is small, this system is low-dimensional and
computationally efficient.

Finally, the quantities $J_i(\theta)$, $\psi_i(\theta,\mu)$, and
$H_i(\theta,\mu)$ can be computed efficiently using automatic
differentiation.
