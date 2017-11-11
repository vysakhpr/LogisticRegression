import subprocess
import string 
import time
import math
import operator
import numpy as np

months=["january","february", "march","april","may","june","july","august","september","october","november","december"]
stopwords=["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]

def process(filename):
	word_dict={}
	label_dict={}
	label_len=0
	word_len=0
	cat=subprocess.Popen(["hadoop", "dfs", "-cat", filename], stdout=subprocess.PIPE)
	for line in cat.stdout:
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
		parts=line.split("\t")
		labels=parts[0].replace(" ","").split(',')
		document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
		words=filter(None,document.split(" "))
		words=[word for  word in words if word.lower() not in months]
		words=[word for  word in words if word.lower() not in stopwords]
		
	cat=subprocess.Popen(["hadoop", "fs", "-cat", "/user/ds222/assignment-1/DBPedia.verysmall/verysmall_train.txt"], stdout=subprocess.PIPE)
	buff=""
	k=0
	p=0
	for line in cat.stdout:
		if "\t" not in line:
			continue
		parts=line.split("\t")
		labels=parts[0].replace(" ","").split(',')
		document=parts[1].split('"',1)[1].rsplit('"',1)[0].replace("\\u","").translate(None,string.punctuation).translate(None,string.digits)
		words=filter(None,document.split(" "))
		words=[word for  word in words if word.lower() not in months]
		words=[word for  word in words if word.lower() not in stopwords]
		s=""
		s=s+str(label_dict[labels[0]])+" "
		f=["0"]*len(word_dict)
		for word in words:
			if word in word_dict:
				f[word_dict[word]]="1"
		s=s+" ".join(f)+"\n"
		k=k+1
		with open("output.txt","a") as mf:
			mf.write(s)
		if(k==5000):
			x=["hadoop", "fs", "-put","output.txt", "/user/vysakhr/inputfile/part"+str(p)+".txt"]
			subprocess.call(x)
			subprocess.call("rm output.txt",shell=True)
			p=p+1
			k=0

	if s!= "":
		x=["hadoop", "fs", "-put","output.txt", "/user/vysakhr/inputfile/part"+str(p)+".txt"]
		subprocess.call(x)
		subprocess.call("rm output.txt",shell=True)
	print len(word_dict),p
process("/user/ds222/assignment-1/DBPedia.verysmall/verysmall_train.txt")