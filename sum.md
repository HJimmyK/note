### 第一章：基本概念
#### 事件与集合
**随机试验**：在相同条件下可以重复；结果不止一个，事先知道所有可能结果；每次试验无法预测结果

**事件**：对于随机试验，把每一个可能的结果称为事件

**随机事件**：可能发生也可能不发生的事件

**基本事件(样本点)**：相对于实验目的，不可再分的事件

**复合事件**：由基本事件构成

**样本空间(或必然事件**)：所有样本点构成的集合，记作 $\Omega$ 

**不可能事件**：不含任何样本点，记作 $\varnothing$

事件的本质是集合。自然地，我们会想到集合间的运算，常见集合运算如下图
![alt text](image.png)

**交换律**：$A\cup B=B\cup A,~A\cap B=B\cap A$
**结合律**：$\\A\cup(B\cup C)=(A\cup B)\cup C,\\A(BC)=(AB)C$
**分配律**：$A(B\cup C)=(AB)\cup(AC),\\
(AB)\cup C=(A\cup C)(B\cup C),A(B-C)=AB-AC$
**德摩根律**：$\overline{A\cup B}=\overline{A}\cap\overline{B},\overline{A\cap B}=\overline{A}\cup\overline{B}$

**事件的积**：$A\cap B=AB$
**事件的和**：$A\cup B$
**事件的差**：$A-B=A\Omega-AB=A\overline{B}$

<div class="custom-div-warning">
    <strong>互不相容与对立的区别与联系：</strong><br> 
    1. 两事件对立，一定互不相容<br>
    2. 互不相容适用于多个事件；对立只适用于两个事件<br>
    3. 互不相容不能同时发生，可都不发生；对立，有且只有一个发生
</div>

**完备事件组**：若 $A_1,A_2,\cdots,A_n$ 两两互不相容，且 $\cup_{i=1}^{n}A_i=\Omega$


<div class="custom-div-note">

**常见事件**：<br>
$A$发生：$A$
只有$A$发生：$A\overline BC$
$A$、$B$、$C$恰有一个发生：$A\overline{B}\overline{C}+\overline{A}B\overline{C}+\overline{A}\overline{B}C$
$A、B、C$同时发生：$ABC$
$A、B、C$至少有一个发生：$A+B+C$
$A、B、C$至多一个发生：$\overline {A}\overline {B}\overline {C}+ A\overline {B}\overline {C}+ \overline {A}B\overline {C}+ \overline {A}\overline {B}C$
恰有两个发生：$\overline{A}BC+A\overline{B}C+AB\overline{C}$
至少有两个发生：$AB+BC+AC$
$A$发生必然导致$B$发生，则称$A\subset B;A$发生必然导致$B$不发生，则称$A\subset\overline{B};$
</div>

**加法公式**：
$$
对于任意两事件A,B，有\\
P(A\cup B)=P(A)+P(B)-P(AB)\qquad\quad\tag{表达式1-1}
$$

$$
对于n个事件A_1,A_2,\cdots,A_n,可用数学归纳法证得：\\
\begin{align*}
    P(A_1\cup A_2\cup\cdots\cup A_n)&=\sum_{i=1}^nP(A_i)-\sum_{1\le i<j\leqslant n}P(A_iA_j)\\
    &\quad+\sum_{1\leqslant i<j<k\leqslant n}P(A_iA_jA_k)+\cdots+\\
    &\quad+(-1)^{n-1}P(A_1A_2\cdots A_n)\tag{表达式1-2}
\end{align*}
$$

#### 条件概率
设事件 $A,B$ 是两个事件，且 $P(A)>0$，称
$$
P(B|A)=\frac{P(AB)}{P(A)}\tag{定义1-1}
$$
为在事件 $A$ 发生的前提下事件 $B$ 发生的**条件概率**。

**乘法公式**：
$$
设P(A)>0,则有\\
P(AB)=P(B|A)P(A)\tag{表达式1-3}
$$

$$
一般的，设A_1,A_2,\cdots,A_n为n个事件，n\ge2，\\
且P(A_1A_2\cdots A_{n-1})>0，则有\\
\begin{align*}
    P(A_1A_2\cdots A_{n-1})&=P(A_n|A_1A_2\cdots A_{n-1})P(A_{n-1}A_1A_2\cdots A_{n-2})\\
    &\quad\cdots P(A_2|A_1)P(A_1)\tag{表达式1-4}
\end{align*}
$$

**全概率公式**：
$$
P(B)=\sum_{i=1}^nP(A_iB)=\sum_{i=1}^nP(A_i)P(B|A_i)\tag{定理1-1}
$$
**$Bayes$公式**:

$$
P(A_i|B)=\frac{P(A_iB)}{P(B)}=\frac{P(A_i)P(B|Ai)}{\sum\limits_{i=1}^nP(A_i)P(B|A_i)}\tag{定理1-2}
$$

#### 独立性
设事件 $A,B$ 是两个事件，如果满足等式
$$
P(AB)=P(B)P(A)\tag{定义1-2}
$$
为称事件 $A,B$ **相互独立**，简称 $A,B$ **独立**。



### 第二章：(多维)随机变量及其分布

#### 概念
##### 随机变量
设随机变量的样本空间为 $S=\{e\}$. $X=X(e)$ 是定义在样本空间 $S$ 上的实值单值函数. 称 $X=X(e)$ 为**随机变量**.(定义 2-1)

##### 分布函数
设 $X$ 是一个随机变量，$x$ 是任意实数，函数
$$
F(x)=P\{X\leqslant x\},-\infty<x<\infty\tag{定义2-1}
$$
称为 $X$ 的**分布函数**。

对于任意实数 $x_1,x_2(x_1<x_2)$，有
$$
P\{x_1<X\leqslant x_2\}=P\{X\leqslant x_2\}-P\{X\leqslant x_1\}=F(x_2)-F(x_1)
$$
因此，若已知$X$的分布函数，我们就知道$X$落在任一区间$(x_1,x_2]$上的概率，从这个意义上说，分布函数完整地描述了随机变量的统计规律性。

分布函数具有以下性质：
1. 有界性。 $0\leqslant F(x)\leqslant 1$
2. 单调性。$F(x)$ 是一个不减函数
3. 右连续。$F(x+0)=F(x)$

##### 概率密度函数
如果对于随机变量 $X$ 的分布函数 $F(x)$，存在非负可积函数 $f(x)$，使对于任意实数 $x$ 有
$$
F(x)=\int_{-\infty}^xf(x)\text{d}x\tag{定义2-2}
$$
则称 $f(x)$ 为 $X$ 的**概率密度函数**，简称**概率密度**。

概率密度函数具有以下性质：
1. $f(x)\geqslant 0$
2. $\int_{-\infty}^{\infty}f(z)\text{d}x=1$
3. 对于任意实数 $x_1,x_2(x_1\leqslant x_2)$，有$$P\{x_1<X\leqslant x_2\}=F(x_1)-F(x_2)=\int_{x_1}^{x_2}f(x)\text{d}x\tag{性质2-1}$$
4. 若 $f(x)$ 在点 $x$ 处连续，则有 $F'(x)=f(x)$

##### 二维随机变量
一般，设 $E$ 是一个随机变量，他的样本空间为 $S=\{e\}$. $X=X(e)$ 和 $Y=Y(e)$ 是定义在样本空间 $S$ 上的随机变量，由他们组成一个向量 $(X,Y)$。称为**二维随机向量**或**二维随机变量**。(定义 2-3)

##### 联合分布函数
设 $(X,Y)$ 是一个随机变量，$x,y$ 是任意实数，二元函数
$$
F(x,y)=P\{(X\leqslant x)\cap(Y\leqslant y)\}\tag{定义2-4}
$$
记成 $P\{X\leqslant x,Y\leqslant y\}$
称为二维随机变量 $(X,Y)$ 的**分布函数**，或称为随机变量 $X,Y$ 的**联合分布函数**。

其具有以下性质：
1. 有界性。$0\leqslant F(x,y)\leqslant 1$
2. 单调性。$F(x,y)$ 关于 $x$ 和 $y$ 分别是非减函数
3. 函数极限
$\begin{gathered}\lim_{x\to-\infty}F(x,y)=F(-\infty,-\infty)=0 \\\lim_{x\to-\infty}F(x,y)=F(-\infty,y)=0 \\\lim_{y\to\infty}F(x,y)=F(x,-\infty)=0 \\F(+\infty,+\infty)=1 \end{gathered}$
4. $F(x,y)$ 关于每个变元是右连续的
5. 对于任意实数 $x_1,x_2(x_1\leqslant x_2)$ 和 $y_1,y_2(y_1\leqslant y_2)$ 有 $\begin{aligned}&P\{x_1<X\leqslant x_2,y_1<Y\leqslant y_2\}=F(x_2,y_2)-F(x_2,y_1)-F(x_1,y_2)+F(x_1,y_1)\end{aligned}$

##### 联合概率密度函数
与一维随机变量相似，对于二维随机变量 $(X,Y)$ 的分布函数 $F(x,y)$，存在非负可积函数 $f(x,y)$，使对于任意实数 $x,y$ 有
$$
F(x,y)=\int_{-\infty}^x\int_{-\infty}^yf(u,v)\text{d}u\text{d}v\tag{定义2-5}
$$
则称 $f(x,y)$ 为二维随机变量 $(X,Y)$ 的**概率密度函数**，或称为随机变量 $X$ 和 $Y$ 的**联合概率密度**。

联合概率密度具有以下性质：
1. $f(x,y)\geqslant 0$
2. $\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}f(u,v)\text{d}u\text{d}v=F(-\infty,\infty)=1$
3. 设 $G$ 为 $xOy$ 平面上的区域，点 $(X,Y)$ 落在 $G$ 内的概率为 $$P\{(X,Y)\in G\}=\iint\limits_{G}f(x,y)\text{d}x\text{d}y\tag{性质2-2}$$
4. 若 $f(x,y)$ 在点 $(x,y)$ 处连续，则有 $\frac{\partial^2 F(x,y)}{\partial x\partial y}=f(x,y)$

##### 边缘分布
二维随机变量 $(X,Y)$ 作为一个整体，具有一个分布函数 $F(x,y)$，而 $X$ 和 $Y$ 都是随机变量，各自也有分布函数，将他们分别记作 $F_X(x),F_Y(y)$，依次称为二维随机变量 $(X,Y)$ 关于 $X$ 和关于 $Y$ 的**边缘分布函数**。边缘分布函数可以由 $(X,Y)$ 的分布函数 $F(x,y)$ 所确定，事实上，
$$
F_X(x)=P\{X\leqslant x\}=P\{X\leqslant x,Y<\infty\}=F(x,\infty)\tag{定义2-6}
$$

对于离散型
$$
p_i=\sum_jp_{ij}\\p_j=\sum_ip_{ij}\tag{表达式2-1}
$$

对于连续型
$$
\frac{\partial F}{\partial x}=f_X(x)=\int_{-\infty}^{+\infty}f(x,y)\text{d}y\\\frac{\partial F}{\partial y}=f_Y(y)=\int_{-\infty}^{+\infty}f(x,y)\text{d}x\tag{表达式2-2}
$$

##### 条件分布
设 $(X,Y)$ 是二维离散随机变量，对于固定的 $j$，若 $P\{Y=y_j\}>0$，则称
$$
P\{X=x_i|Y=y_j\}=\\\frac{P\{X=x_i,Y=y_j\}}{P\{Y=y_j\}}=\frac{p_{ij}}{p_{\cdot j}},i=1,2,\cdots\tag{定义2-6}
$$
为在 $Y=y_j$ 条件下随机变量 $X$ 的**条件分布律**。
同样，对于固定的 $i$，若 $P\{X=x_i\}>0$，则称
$$
P\{Y=y_j|X=x_i\}=\\\frac{P\{X=x_i,Y=y_j\}}{P\{X=x_i\}}=\frac{p_{ij}}{p_{i\cdot}},j=1,2,\cdots\tag{定义2-7}
$$
为在 $X=x_j$ 条件下随机变量 $Y$ 的**条件分布律**。


设二维随机变量 $(X,Y)$ 的概率密度函数为 $f(x,y)$，$(X,Y)$ 关于 $Y$ 的边缘概率密度函数为 $f_Y(y)$。若对于固定的 $y$，$f_Y(y)>0$，则称 $\frac{f(x,y)}{f_Y(y)}$ 为在 $Y=y$ 的条件下 $X$ 的条件概率密度，记为
$$
f_{X|Y}(x|y)=\frac{f(x,y)}{f_Y(y)}\tag{定义2-8}
$$
称 $\int_{-\infty}^xf_{X|Y}(x|y)\text{d}x=\int_{-\infty}^x\frac{f(x,y)}{f_Y(y)}\text{d}x$ 为在 $Y=y$ 的条件下 $X$ 的**条件分布函数**，记为 $P\{X\leqslant x|Y=y\}$ 或者 $F_{X|Y}(x|y)$，即
$$
F_{X|Y}(x|y)=P\{X\leqslant x|Y=y\}=\int_{-\infty}^x\frac{f(x,y)}{f_Y(y)}\text{d}x\tag{定义2-9}
$$

<div class="custom-div-note">

<strong>条件概率密度函数的形式由来</strong>

考虑条件概率
$$
P\{\left.X\leqslant x \right| y<Y\leqslant y+\epsilon\}
$$
设 $P\{y<Y\leqslant y+\epsilon\}>0$，则有
$$
\begin{align*}
    P\{\left.X\leqslant x \right| y<Y\leqslant y+\epsilon\}&=\frac{ P\{X\leqslant x,y<Y\leqslant y+\epsilon\}}{P\{y<Y\leqslant y+\epsilon\}}\\
    &=\frac{\int_{-\infty}^x\begin{bmatrix*}\int_{y}^{y+\epsilon}f(x,y)\text{d}y\end{bmatrix*}\text{d}x}{\int_{y}^{y+\epsilon}f_Y(y)\text{d}y}
\end{align*}
$$
$\epsilon$ 和 $\text{d}y$ 都是小量，那么我们可以和容易的知道，在区间 $(y,y+\epsilon]$ 内，函数的变化相当的小，可以用矩形面积逼近，那么
$$
\int_{y}^{y+\epsilon}f(x,y)\text{d}y=f(x,y)\epsilon\\
\int_{y}^{y+\epsilon}f_Y(y)\text{d}y=f_Y(y)\epsilon
$$
那么
$$
\begin{align*}
    P\{\left.X\leqslant x \right| y<Y\leqslant y+\epsilon\}&=\frac{\int_{-\infty}^xf(x,y)\epsilon\text{d}x}{f_Y(y)\epsilon}\\
    &=\int_{-\infty}^x\frac{f(x,y)}{f_Y(y)}\text{d}x
\end{align*}
$$
</div>

##### 相互独立的随机变量
设 $F(x,y)$ 及 $F_X(x),F_Y(y)$ 分别是二维随机变量 $(X,Y)$ 的分布函数及边缘分布函数。若对于所有的 $x,y$，都有
$$
P\{X\leqslant x,Y\leqslant y\}=P\{(X\leqslant x)\}P\{(Y\leqslant y)\}\tag{定义2-10}
$$
即
$$
F(x,y)=F_X(x)F_Y(y)\tag{定义2-11}
$$
则称随机变量 $X$ 和 $Y$ 是**相互独立的**。

如果 $(X,Y)$ 是连续型随机变量，那么 $X$ 和 $Y$ 独立的等价条件是
$$
f(x,y)=f(x)f(y)\tag{定义2-11-2}
$$
在平面上“几乎处处成立”。
如果 $(X,Y)$ 是离散型随机变量，那么 $X$ 和 $Y$ 独立的等价条件是：对于 $(X,Y)$ 的所有可能取得值 $(x_i,y_i)$ 有
$$
P\{X=x_i,Y=y_i\}=P\{X=x_i\}P\{Y=y_i\}\tag{定义2-10-2}
$$

#### n维随机变量的推广
以上所说的关于二维的随机变量的一些概念，容易推广到 $n$ 维随机变量的情况。

$n$ 维随机变量 $(X_1,X_2,\cdots,X_n)$ 的**分布函数**定义为
$$
F(x_1,x_2,\cdots,x_n)=P\{X_1\leqslant x_1,X_2\leqslant x_2,\cdots,X_n\leqslant x_n\}\\\tag{定义2-12}
$$
其中，$x_1,x_2,\cdots,x_n$ 为任意实数。

若存在非负可积函数 $f(x_1,x_2,\cdots,x_n)$，使得对于任意实数 $x_1,x_2,\cdots,x_n$ 有
$$
F(x_1,x_2,\cdots,x_n)=\int_{-\infty}^{x_n}\int_{-\infty}^{x_{n-1}}\cdots\int_{-\infty}^{x_1}f(x_1,x_2,\cdots,x_n)\text{d}x_1\text{d}x_2\cdots\text{d}x_n\\\tag{定义2-13}
$$
则称 $f(x_1,x_2,\cdots,x_n)$ 为 $(X_1,X_2,\cdots,X_n)$ 的**概率密度函数**。

若随机变量 $(X_1,X_2,\cdots,X_n)$ 的**分布函数**为 $F(x_1,x_2,\cdots,x_n)$，则 $(X_1,X_2,\cdots,X_n)$ 的 $k(1\leqslant k<n)$ 维边缘分布函数就随之确定。例如 $(X_1,X_2,\cdots,X_n)$ 关于 $X_1$，关于 $(X_1,X_2)$ 的边缘分布函数分别为
$$
F_{X_1}(x_1)=F(x_1,\infty,\infty,\cdots,\infty)\\
F_{X_1,X_2}=F(x_1,x_2,\infty,\cdots,\infty)
$$
边缘概率密度函数分别为
$$
f_{X_1}(x_1)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}f(x_1,x_2,\cdots,x_n)\text{d}x_2\text{d}x_3\cdots\text{d}x_n\\
f_{X_1,X_2}(x_1,x_2)=\int_{-\infty}^{\infty}\int_{-\infty}^{\infty}\cdots\int_{-\infty}^{\infty}f(x_1,x_2,\cdots,x_n)\text{d}x_3\text{d}x_4\cdots\text{d}x_n
$$
若对于所有 $x_1,x_2,\cdots,x_n$ 都有
$$
F(x_1,x_2,\cdots,x_n)=F_{X_1}(x_1)F_{X_2}(x_2)\cdots F_{X_n}(x_1)\tag{定义2-14}
$$
则称 $X_1,X_2,\cdots,X_n$ 是相互独立的。
若对于所有 $x_1,x_2,\cdots,x_m,y_1,y_2,\cdots,y_n$，有
$$
F(x_1,x_2,\cdots,x_m，y_1,y_2,\cdots,y_n)=F(x_1，x_2,\cdots,x_m)F(y_1,y_2,\cdots,y_n)\\
\tag{定义2-15}
$$



### 第三章：随机变量的数字特征
#### 期望与方差
设离散型随机变量 $X$ 的分布律为
$$
P\{X_k=x_k\}=p_k,\qquad k=1,2,\cdots
$$
若级数
$$
\sum_{k=1}^nx_kp_k
$$
绝对收敛，则级数 $\sum\limits_{k=1}^nx_kp_k$ 的和称为随机变量的**数学期望**，记为 $E(X)$，即
$$
E(X)=\sum_{k=1}^nx_kp_k\tag{定义3-1}
$$

设连续型随机变量 $X$ 的概率密度为 $F(x)$，若积分
$$
\int_{\infty}^{\infty}xf(x)\text{d}x
$$
绝对收敛，则积分 $\int_{\infty}^{\infty}xf(x)\text{d}x$ 的值称为随机变量 $X$ 的**数学期望**，记为 $E(X)$，即
$$
E(X)=\int_{\infty}^{\infty}xf(x)\text{d}x\tag{定义3-2}
$$

数学分布简称**期望**，又称为**均值**。

设 $X$ 是随机变量，若 $E\{[X-E(X)]^2\}$ 存在，则称其为 $X$ 的**方差**，记为 $D(X)$ 或者 $Var(X)$，即
$$
D(X)=Var(X)=E\{[X-E(X)]^2\}\tag{定义3-3}
$$




### 第四章：常见分布及其性质

### 第五章：大数定理及中心极限定理



### 第六章：样本及抽样分布

##### **定义**：
**样本均值**：
$$
\overline{X}=\frac{1}{n}\sum_{i=1}^{n}X_i\tag{定义6-1}
$$

**样本方差**：
$$
S^2=\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2=\frac{1}{n-1}\begin{pmatrix*}
    \sum\limits_{i=1}^{n-1}X_i^2-n\cdot \overline{X}^2
\end{pmatrix*}\tag{定义6-2}
$$

**样本标准差**：
$$
S=\sqrt{s^2}=\sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2}\tag{定义6-3}
$$

**样本 $k$阶（原点）矩**：
$$
A_k=\frac{1}{n}\sum_{i=1}^{n}X_i^k,\qquad k=1,2,3,\cdots\tag{定义6-4}
$$

**样本 $k$阶中心距**：
$$
B_k=\frac{1}{n}\sum_{i=1}^{n}(X_i-\overline{X})^k,\qquad k=2,3,\cdots\tag{定义6-5}
$$

##### 一些分布

**(1). $\chi^2$分布**
设 $X_1,X_2,\cdots,X_n$ 是来自总体 $N(0,1)$ 的样本，则称统计量
$$
\chi^2=X_1^2+X_2^2+\cdots+X_n^2\tag{定义6-6}
$$
服从自由度为 $n$ 的 $\chi^2$分布，记为 $\chi^2\sim \chi^2(n)$
我们由 $\Gamma$分布定义可知，$\chi^2(1)$分布即 $\Gamma(\frac{1}{2},2)$分布。由 $X_1,X_2,\cdots,X_n$ 的独立性可知 $X_1^2,X_2^2,\cdots,X_n^2$ 也独立，从而由 $\Gamma$分布的可加性可知
$$
\chi^2=X_1^2+X_2^2+\cdots+X_n^2\sim\Gamma(\frac{n}{2},2)
$$
那么
$\chi^2(n)$ 的概率密度为
$$ 
f(x)= \begin{cases}\frac{1}{2^{\frac{n}{2}}\Gamma(\frac{n}{2})}x^{\frac{n}{2}-1}e^{\frac{-x}{x}},& x>0\\0, & \text{其他} \end{cases} \tag{表达式6-1}$$
同样，由 $\Gamma$分布的可加性可知，$\chi^2$分布也具有可加性：
设 $\chi^2_1\sim\chi^2(n_1)$，$\chi^2_2\sim\chi^2(n_2)$，并且 $\chi^2_1,\chi^2_2$ 相互独立，则有
$$
\chi^2_1+\chi^2_2\sim\chi^2(n_1+n_2)\tag{性质6-1}
$$

**(2).$t$分布**
设 $X\sim N(0,1)$，$Y\sim \chi^2(n)$，且 $X,Y$ 相互独立，则称随机变量
$$
t=\frac{X}{\sqrt{Y/n}}\tag{定义6-7}
$$
服从自由度为 $n$ 的 $t$分布，记为 $t\sim t(n)$

$t$分布又称学生式分布。$t(n)$ 的概率密度函数为
$$
f(x)=\frac{\Gamma(\frac{n+1}{2})}{\sqrt{n\pi}\Gamma(\frac{n}{2})}(1+\frac{x^2}{n})^{-(\frac{n+1}{2})},\quad -\infty<t<\infty\tag{表达式6-2}
$$

**(3).$F$分布**
设 $U\sim \chi^2(n_1)$，$V\sim \chi^2(n_2)$，且 $U,V$ 相互独立，则称随机变量
$$
F=\frac{U/n_1}{V/n_2}\tag{定义6-8}
$$
服从自由度为 $(n_1,n_2)$ 的 $F$分布，记为 $F\sim F(n_1,n_2)$

概率密度函数为
$$
f(x)=\begin{cases}\frac{\Gamma[(n_1+n_2)/2](n_1/n_2)^{n_1/2}x^{(n_1/2)-1}}{\Gamma(n_1/2)\Gamma(n_2/2)[1+(n_1y/n_2)]^{(n_1+n_2)/2}}, &x>0 \\\\ 0 ,&\text{其他} \end{cases}\tag{表达式6-9}
$$


##### 重要定理

设总体 $X$ 的均值为 $\mu$，方差为 $\sigma^2$，$X_1,X_2,\cdots,X_n$ 是来自总体 $X$ 的一个样本，$\overline{X},S^2$ 分别为样本均值和样本方差，则有：
$$
E(\overline{X})=\mu,D(\overline{X})=\frac{\sigma^2}{n}
$$
$$
E(S^2)=\sigma^2\tag{定理6-1}
$$

特别的，当总体 $X\sim N(\mu,\sigma^2)$，有
$$
D(S^2)=\frac{2\sigma^4}{n-1}\tag{定理6-1-2}
$$

设总体 $X$ 的均值为 $\mu$，方差为 $\sigma^2$，$X_1,X_2,\cdots,X_n$ 是来自正态总体 $N(\mu,\sigma^2)$ 的一个样本，$\overline{X},S^2$ 分别为样本均值和样本方差，则有：

$$
\overline{X}\sim N(\mu,\frac{\sigma^2}{n})
$$

$$
\frac{(n-1)S^2}{\sigma^2}\sim \chi^2(n-1)\\
$$

$$
\overline{X}和S^2相互独立\tag{定理6-2}
$$

。。。。。。。。。
。。。。。。。。。
。。。。。。。。。
证明：
(定理6-1)
$$
\begin{align*}
    E(\overline{X})&=E(\frac{1}{n}\sum_{i=1}^{n}X_i)\\
    &=\frac{1}{n}\sum_{i=1}^{n}E(X_i)\\
    &=\frac{1}{n}n\cdot \mu\\
    &=\mu
\end{align*}
$$
$$
\begin{align*}
    D(\overline{X})&=E(\overline{X}^2)-E(\overline{X})^2\\
    &=E(\frac{1}{n^2}\sum_{i=1}^{n}X_i\sum_{j=1}^{n}X_j)-\mu^2\\
\end{align*}
$$
由于 $X_i$ 彼此是独立的，那么 $E(X_iX_j)=\mu^2,(i\ne j)$，且易知 $E(X_i^2)=D(X_i)+E(X_i)^2=\sigma^2+\mu^2$

那么
$$
\begin{align*}
    D(\overline{X})&=\frac{1}{n^2}\sum_{i=1}^{n}E(X_i\sum_{j=1}^{n}X_j)-\mu^2\\
    &=\frac{1}{n^2}[(n^2-n)\mu^2-(\sigma^2+\mu^2)n]-\mu^2\\
    &=\frac{\sigma^2}{n}
\end{align*}
$$

$$
\begin{align*}
    E(S^2)&=E [\frac{1}{n-1}\sum_{i=1}^{n}(X_i-\overline{X})^2]\\
    &=E[\frac{1}{n-1}\begin{pmatrix*}
    \sum\limits_{i=1}^{n}X_i^2-n\cdot \overline{X}^2
    \end{pmatrix*}]\\
    &=\frac{1}{n-1}[(\sigma^2+\mu^2)n-n(\frac{\sigma^2}{n}+\mu^2)]
    \\
    &=\sigma^2
\end{align*}
$$

(定理6-1-2)
令 $Y_i=X_i-\overline{X}$
那么，
$$
S^2=\frac{1}{n-1}\begin{pmatrix*}
    \sum\limits_{i=1}^nY_i^2-n\overline{Y}^2
\end{pmatrix*}
$$
则
$$
\begin{align*}
    D(S^2)&=E((S^2)^2)-E(S^2)^2\\
\end{align*}
$$

$$
\begin{align*}
    E((S^2)^2)&=E [\frac{1}{(n-1)^2}\begin{pmatrix*}
    \sum\limits_{i=1}^nY_i^2-n\overline{Y}^2
    \end{pmatrix*}^2]\\
    (n-1)^2E((S^2)^2)&=E((\sum\limits_{i=1}^nY_i^2)^2)-2nE(\overline{Y}^2\sum\limits_{i=1}^nY_i^2)+n^2E(\overline{Y}^4)\\
    &=E((\sum\limits_{i=1}^nY_i^2)^2)-
\end{align*}
$$




....
，
，
，。。。
。
。

### 附录
分布|	参数|	分布律或概率密度
:---:|:---:|:---:
**(0-1)分布**| $0<p<1$ | $ P\{ X=k\}=p^k(1-p)^{1-k}\\ k=0,1$ 
**二项分布**| $n\geqslant 1$<br> $0<p<1$ | $ P\{ X=k \}= \begin{pmatrix} n\\ k \end{pmatrix} p^k(1-p)^{n-k}\\ k=0,1,2,\cdots  $ 
**负二项分布<br>(帕斯卡分布)**| $ r\geqslant 1 \\ 0<p<1 $ | $ P\{ X=k \}= \begin{pmatrix} k-1\\ r-1 \end{pmatrix} p^k(1-p)^{k-r}\\ k=r,r+1,r+2,\cdots  $
**几何分布** | $0<p<1$ | $ P\{ X=k\}=p(1-p)^{k-1}\\ k=1,2\cdots$ 
**超几何分布**| $N,M,n\\(M\leqslant N)\\(n\leqslant N) $ | $ P\{ X=k \}=\frac{\begin{pmatrix} M\\ k \end{pmatrix}\begin{pmatrix} N-M\\ n-k \end{pmatrix}}{\begin{pmatrix} N\\ n \end{pmatrix}} \\ k为整数，max\{0,n-N+M\}\leqslant k\leqslant \min\{n,M\} $ 
**泊松分布**| $\lambda>0$ | $$P\{X=k\} = \frac{\lambda ^k e^{-k}}{k!}\\k=0,1,2,\cdots$$
**均匀分布**| $a<b$ | $f(x)=\begin{cases}\frac 1 {b-a}, & a< x < π\\0,& \text{其他}\end{cases}$
**正态分布**| $\mu\\\sigma>0$ | $$ f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{\frac{-(x-\mu)^2}{2\sigma^2}} $$
**$\Gamma$ 分布** | $\alpha>0\\\beta>0$ | $$ f(x)= \begin{cases} \frac{1}{\beta^{\alpha}\Gamma(\alpha)}x^{\alpha-1}e^{\frac{1}{\beta}},& x>0\\0, & \text{其他} \end{cases} $$
**指数分布<br>(负指数分布)** | $\theta>0$ | $$ f(x)=\begin{cases}\frac{1}{\theta} e^{\frac{-x}{\beta}}, & x>0\\0,& \text{其他} \end{cases} $$ 
**卡方分布<br>( $\mathcal{X}^2$ 分布)** | $n\geqslant 0$ | $$ f(x)= \begin{cases}\frac{1}{2^{\frac{n}{2}}\Gamma(\frac{n}{2})}x^{\frac{n}{2}-1}e^{\frac{-x}{x}},& x>0\\0, & \text{其他} \end{cases} $$
**威布尔分布** | $\eta>0\\\beta>0$ | $$ f(x)= \begin{cases}\frac{\beta}{\eta}(\frac{x}{\eta})^{\beta-1}e^{-(\frac{x}{\eta})^{\beta}},& x>0\\0, & \text{其他} \end{cases}$$
**瑞利分布** | $\sigma>0$ | $$ f(x)=\begin{cases} \frac{x}{\sigma^2}e^{-\frac{x^2}{2\sigma^2}}, & x>0\\ 0, & \text{其他}\end{cases}$$
**$\beta$ 分布** | $\alpha>0\\\beta>0$ | $$ f(x)=\begin{cases} \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)}x^{\alpha-1}(1-x)^{\beta-1}, & 0<x<1 \\ 0, & \text{其他}\end{cases}$$
**对数<br>正态分布**| $\mu\\\sigma>0$ | $$ f(x)=\begin{cases}\frac{1}{x\sigma \sqrt{2\pi}}e^{-\frac{(\text{ln}x-\mu)^2}{2\sigma^2}}, & x>0\\0, & \text{其他} \end{cases}$$
**柯西分布** | $a\\\lambda>0$ | $$f(x)=\frac{1}{\pi}\frac{\lambda}{\lambda^2+(x-a)^2}$$
**$t$ 分布**| $n\geqslant 1$ | $$f(x)=\frac{\Gamma(\frac{n+1}{2})}{\sqrt{n\pi}\Gamma(\frac{n}{2})}(1+\frac{x^2}{n})^{-(\frac{n+1}{2})} $$
**$F$ 分布**| $n_1,n_2$ | $$f(x)=\begin{cases}\frac{\Gamma[(n_1+n_2)/2](n_1/n_2)^{n_1/2}x^{(n_1/2)-1}}{\Gamma(n_1/2)\Gamma(n_2/2)[1+(n_1y/n_2)]^{(n_1+n_2)/2}}, &x>0 \\\\ 0 ,&\text{其他} \end{cases}$$

aa



