import numpy as np 
import time

class Approx():

	def __init__(self, A, k):
		"""
		A is the full matrix.
		k determines the number of columns to be sampled.
		matrices sampled from A will be denoted with the list of
		indices of their columns wrt A.
		delta determines failure probability.
		"""
		self.A = A
		self.k = k
		# self.c1 = 1.0 
		# self.delta = 0.1
		self.size = 3 # determines the sample size

	def RidgeLeverageScoreApprox(self, A, A_half, w):
		"""
		Inputs:
		A is the full matrix at this recursive level.
		A_half is the matrix wrt which the leverage score is computed.

		Outputs:
		A_tilde is the sampled matrix based on ridge leverage scores.
		w_tilde are the wieights of columns in A_tilde.
		"""
		tau = []
		p = []

		# compute top k approx of A_half
		if w is None:
			A_half_wt = self.A[:, A_half]
		else:
			A_half_wt = np.matmul(self.A[:, A_half], np.diag(w)) # weighing A_half
		u, s, v = np.linalg.svd(A_half_wt, full_matrices=False)
		u = u[:, : self.k]
		v = v[: self.k, :]
		s = s[: self.k]
		A_half_k = np.dot(np.dot(u, np.diag(s)), v)

		# ridge term
		Lambda = np.linalg.norm((A_half_wt - A_half_k), ord="fro")**2/self.k 
		M = np.matmul(A_half_wt, A_half_wt.T)
		M = M + np.identity(len(M))*Lambda

		# compute the ridge leverage scores and probabilities for sampling
		for i in range(len(A)):
			tau.append(int(np.matmul(self.A[:, A[i]].T, np.matmul(M, self.A[:, A[i]]))))
			# p.append(np.min(1, tau[-1]*self.c1*np.log10(self.k/self.delta)))
		p = np.array(tau)
		p = p/p.sum()

		i_A_tilde = np.random.choice(len(A), size=len(A_half), replace=False, p=p)
		A_tilde = A[i_A_tilde] # samplesm A_tilde from A based on ridge leverage score on A_half
		w_tilde = np.reciprocal(np.array(p[i_A_tilde]))**0.5 # weights of columns in A_tilde
		return A_tilde, w_tilde

	def RepeatedHalving(self, A, n=1):
		"""
		A is the full matrix at this recursive level.
		n is the recursive level.
		"""
		A_half = np.random.choice(A, size=int(len(A)*0.5), replace=False) #uniformly sample half the columns
		
		# check recursion end
		# if len(A_half) <= self.k*np.log10(self.k/self.delta):
		if len(A_half) <= self.k*self.size:
			return self.RidgeLeverageScoreApprox(A, A_half, None)

		else:
			A_tilde, w_tilde = self.RepeatedHalving(A_half, n+1)
			return self.RidgeLeverageScoreApprox(A, A_tilde, w_tilde)

	def getApprox(self):
		# recursively compute the sparse time matrix approximate 
		# with sampling with ridge leverage score.
		A = np.arange(self.A.shape[1])
		A_tilde, w_tilde = self.RepeatedHalving(A)
		self.A_tilde = A_tilde
		self.w_tilde = w_tilde
		return np.matmul(self.A[:, A_tilde], np.diag(w_tilde))

start = time.time()
A = np.matrix(np.random.rand(512, 50000))
k = 1000
Appr = Approx(A, k)
weighted_A_tilde = Appr.getApprox()
print(weighted_A_tilde.shape, time.time()-start)