from Rohan_Ayush_A1 import build_codewords, build_huffman_tree,Alphabet


#this will generate freqency of all the characters in freq.txt 
def generate_freq(sequence):
	freq_count = dict()
	for ele in sequence:
		if ele in freq_count:
			freq_count[ele] += 1
		else:
			freq_count[ele] = 1




#Creating a list of alphabet objects
for ele in freq_count:
	freq.append(Alphabet(ele , freq_count[ele]))


#Building the huffman tree from the list of alphabet objects
root = build_huffman_tree(freq)
print("Huffman tree built")

#Creating an empty dictionary for storing the codewords
codewords = dict()

#Building codewords from the huffman tree that we built and storing them in codewords dictionary
build_codewords("" , root,codewords)

#Writing the codewords corresponding to each character in a file(codewords.txt)
result = open('codewords.txt', 'w')
for i in freq:
	if(i.get_alpha() in codewords):
		add = i.get_alpha() + " " + codewords[i.get_alpha()] + "\n"
		result.write(add)
result.close()
print("Codewords generated in codewords.txt")

#Encoding the the file(sample.txt) using the codewords generated and writing it to encoded_text.txt
file_out = open("encoded_text.txt", "w")
while True:
	char = file.read(1)
	if(not char):
		file_out.write(codewords["EOM"])
		break
	char = getchar(char)
	file_out.write(codewords[char])

print("encoding of sample.txt done in encoded_text.txt")
file_out.close()
file.close()