from random import randint
import string 
import time
import math
import operator
threshold=5
months=["january","february", "march","april","may","june","july","august","september","october","november","december"]
stopwords=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]



def random_sample(filename):
	with open(filename) as f:
		for i, l in enumerate(f):
			pass
	l=i+1
	n=randint(0,l-1)
	i=0
	f=open(filename,'r')
	for line in f:
		if "\t" not in line:
			continue
		if i==n:
			parts=line.split("\t")
			labels=parts[0].replace(" ","").split(',')
			document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
			words=filter(None,document.split(" "))
			words=[word for  word in words if word.lower() not in months]
			words=[word for  word in words if word.lower() not in stopwords]	
			return words,labels
		i=i+1
	return [],[]


def data_preprocess(filename):
	word_dict={}
	label_dict={}
	label_len=0
	word_len=0
	f=open(filename,'r')
	for line in f:
		if "\t" not in line:
			continue
		parts=line.split("\t")
		labels=parts[0].replace(" ","").split(',')
		document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
		words=filter(None,document.split(" "))
		words=[word for  word in words if word.lower() not in months]
		words=[word for  word in words if word.lower() not in stopwords]
		for word in words:
			if word not in word_dict:
				word_dict[word]=word_len
				word_len=word_len+1
			# else:
			# 	word_dict[word]=word_dict[word]+1	
		for label in labels:
			if label not in label_dict:
				label_dict[label]=label_len
				label_len=label_len+1

	return word_dict,label_dict
def sum(w1,w2):
	w=[0]*len(w1)
	for i in range(len(w1)):
		w[i]=w1[i]+w2[i]
	return w

def product(w1,c):
	w=[0]*len(w1)
	for i in range(len(w1)):
		w[i]=w1[i]*c
	return w

def dot(w1,w2):
	s=0
	for i in range(len(w1)):
		s=s+w1[i]*w2[i]
	return s

def sigmoid(x):
	return 1.0/(1+math.exp(-1*x))

def vec_sum(x):
	s=0
	for i in range(len(x)):
		s=s+x[i]
	return s

def norm_diff(w1,w2):
	s=0
	for i in range(len(w1)):
		s=s+math.pow(w1[i]-w2[i],2)
	return math.sqrt(s)



def update(x,y,label_dict,w,i,eta):
	lbda=0.001
	# for label in label_dict:
	# 	if label_dict[label]==i:
	# 		current_label=label
	# 		break
	l=0
	for label in y:
		if label_dict[label]==i:
			l=1
	c=eta*l*sigmoid(dot(w,x))
	w1=w
	w=sum(w,product(x,c))
	w=sum(w,product(w1,-2*lbda))
	return w

features,label_dict=data_preprocess("/scratch/ds222-2017/assignment-1/DBPedia.small/small_training.txt")
n=len(features)
m=len(label_dict)
w0=[[0]*(n+1)]*m
w1=[[0]*(n+1)]*m
eta=0.01
gamma=0.001
k=0
g=0
update_vec=[0]*m
while True:
	flag=0
	[words,labels]=random_sample("/scratch/ds222-2017/assignment-1/DBPedia.small/small_training.txt")
	x=[0]*(n+1)
	x[n]=1
	for word in words:
		x[features[word]]=1

	for i in range(m):
		w1[i]=update(x,labels,label_dict,w0[i],i,eta)

	# 	if update_vec[i]==0:
	# 		w1[i]=update(x,labels,label_dict,w0[i],i,eta)
	# 		for label in labels:
	# 			if label_dict[label]==i:
	# 				if i==10:
	# 					print norm_diff(w1[i],w0[i]) 
	# 				if norm_diff(w1[i],w0[i])<gamma:
	# 					update_vec[i]=1
	# 					g=g+1
	# 					print g
	# for i in range(m):
	# 	if update_vec[i]==0:
	# 		k=k+1
	# 		flag=1
	# 		break
	# if flag==0:
	# 	break
	# print g
	g=g+1
	if g>1000:
		break;
for i in range(len(w1)):
	for j in range(len(w1[0])):
		w0[i][j]=w1[i][j]

print "Finished Training"
with open("/scratch/ds222-2017/assignment-1/DBPedia.small/small_test.txt") as f:
	for i, l in enumerate(f):
		pass
l=i+1
print l
w=w1
f=open("/scratch/ds222-2017/assignment-1/DBPedia.small/small_test.txt",'r')
# f=open("/scratch/ds222-2017/assignment-1/DBPedia.verysmall/verysmall_test.txt",'r')
pred=[]
g=0
h=0
for line in f:
	g=g+1
	if "\t" not in line:
		continue
	label_probability_dict={}
	parts=line.split("\t")
	labels=parts[0].replace(" ","").split(',')
	document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
	words=filter(None,document.split(" "))
	words=[word for  word in words if word.lower() not in months]
	words=[word for  word in words if word.lower() not in stopwords]
	x=[0]*(n+1)
	x[n]=1
	probs=[0]*m
	for word in words:
		x[features[word]]=1
	for i in range(m):
		probs[i]=sigmoid(dot(w[i],x))
	for label in labels:
		print label_dict[label], probs.index(max(probs))
	


	
