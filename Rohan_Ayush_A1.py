import heapq

#Making a class for building a huffman tree 
#each object of the class has character name(alpha), its frequency(freq), left child(left) 
#and right child(right)
class Alphabet:
	def __init__(self , alpha , freq):
		self.alpha = alpha
		self.freq = freq
		self.left = None
		self.right = None

	def __str__(self):
		return self.alpha + " " + str(self.freq)

	def get_freq(self):
		return self.freq

	def get_alpha(self):
		return self.alpha

	def get_left(self):
		return self.left
		
	def get_right(self):
		return self.right

	def set_right(self,  alphabet):
		self.right = alphabet

	def set_left(self , alphabet):
		self.left = alphabet

	def __lt__(self, other):
		if(self.freq < other.freq):
			return True
		elif(self.freq == other.freq):
			return False


#We build the huffman tree given the freqency list using heaps
def build_huffman_tree(freq1):
	freq = freq1.copy()
	heapq.heapify(freq)

	#we are removeing the all the character with zero frequency, so that it doesn't 
	#involve in building the huffmann tree
	while(freq[0].get_freq() == 0):
		heapq.heappop(freq)
	
	while(len(freq) > 1):
		#getting and popping two elements with the least frquency
		min1 = heapq.heappop(freq)
		min2 = heapq.heappop(freq)

		#combining the elements with the least frquency, and adding it to the heap
		alpha = Alphabet("inode" , min1.get_freq() + min2.get_freq())
		heapq.heappush(freq,alpha)

		alpha.set_left(min1)
		alpha.set_right(min2)
	return freq[0]


#given an empty string(code), root of the huffman tree(root) and an empty  dictionary(codewords) 
#this function will generate the codewords corresponding to the characters
def build_codewords(code , root,codewords):

	leaf = True
	if(root.right != None):
		leaf = False
		build_codewords(code + "1" , root.get_right(),codewords)

	if(root.left != None):
		leaf = False
		build_codewords(code + "0" , root.get_left(),codewords)

	#Since huffman code is a prefix code, all the characters for which code is required 
	#would be the leaf nodes only.
	if(leaf) :
		codewords[root.get_alpha()] = code
