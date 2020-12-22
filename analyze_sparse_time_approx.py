import numpy as np 
from sparse_time_approx import Approx 
from matplotlib import pyplot as plt 
import seaborn as sns
sns.set_theme()

def find_best_k(A, k):
	u, s, v = np.linalg.svd(A, full_matrices=False)
	u = u[:, : k]
	v = v[: k, :]
	s = s[: k]
	A_k = np.dot(np.dot(u, np.diag(s)), v)
	return A_k

def Perf(A, A_, k):
	# || A - (C C^{+} A)_{k} ||_{F}^{2} <= (1 + \epsilon) || A - A_{k} ||_{F}^{2}
	Ak = find_best_k(A, k)
	M = np.matmul(A_, np.linalg.pinv(A_))
	M = np.matmul(M, A)
	M = find_best_k(M, k)
	FrA = np.linalg.norm((A - Ak), ord="fro")**2
	FrC = np.linalg.norm((A - M), ord="fro")**2
	return FrC/(FrA+1e-8) # Finding lower bound for (1 + \epsilon)

P = []
for k in range(10, 101, 10):
	p = 0
	for i in range(100):
		print("k = {}, i = {}".format(k, i), flush=True, end="\r")
		A = np.matrix(np.random.rand(500, 500))
		A_w = Approx(A, k).getApprox()
		p += Perf(A, A_w, k)
	p /= 100
	P.append(p)

plt.figure(figsize=(10, 5))
plt.plot(range(10, 101, 10), P, '--o')
plt.xlabel("k")
plt.ylabel(r"$(1+\epsilon)$")
plt.title(r"Mean $(1+\epsilon)$ values for 500x500 matrix (100 trials)")
plt.savefig("plot_for_k.png")

P = []
for c in range(500, 5001, 500):
	k = c//4
	p = 0
	for i in range(10):
		print("c = {}, i = {}".format(c, i), flush=True, end="\r")
		A = np.matrix(np.random.rand(c//2, c))
		A_w = Approx(A, k).getApprox()
		p += Perf(A, A_w, k)
	p /= 5
	P.append(p)

plt.figure(figsize=(10, 5))
plt.plot(range(500, 10000, 500), P, '--o')
plt.xlabel("# Columns")
plt.ylabel(r"$(1+\epsilon)$")
plt.title(r"Mean $(1+\epsilon)$ values for (C/2)xC matrix (10 trials)")
plt.savefig("plot_for_C.png")