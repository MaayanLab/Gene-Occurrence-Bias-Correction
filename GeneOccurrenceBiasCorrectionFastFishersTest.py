#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from collections import Counter
from scipy import sparse
import copy
import random

# Array contains all unique genes within library (Chea or Encode) and pertubation experiments 
allUniqueGenes=[]

# Dictionary holds all unique chip-seq genes as keys and values are occurrences
ChipseqDict = {}

# Open Chip-seq file to populate dictionary  
occurencesFile = open("ChIP-seqGeneOccurrences.txt", "r")
occurencesFileLines =  occurencesFile.readlines()    
for line in  occurencesFileLines:
    line = line.rstrip("\n")
    fields = line.split("\t")
    gene = fields[0]
    count = int(fields[1])
    ChipseqDict[gene]=count
occurencesFile.close()

# Liklihood dictionary for Chip-seq gene removal
# keys = numbers, values = gene
# If gene1 occurs 10 times, keys #1-10 will have value=gene1
# Gene2 occurs 9 times, keys #11-19 will have value=gene2
liklihoodRemovalDict = {}
nstart=1
for key, value in ChipseqDict.items():
    numbers = list(range(nstart,nstart+value))
    nstart = value+nstart
    for n in numbers:
        liklihoodRemovalDict[n]=key

# Equal liklihood dictionary for Chip-seq gene removal
# Each gene has only 1 key, all gene have same probability of being removed
# 1=gene1, 2=gene2, 3=gene3
equalLiklihoodRemovalDict={}
count = 0
for key,value in ChipseqDict.items():
    count = count+1
    equalLiklihoodRemovalDict[count]=key
           
#################################################################################################################################################

# Retrieve ChEA/Encode gene sets and terms 
libraryTerms = []
libraryGeneSets = []

libraryFile = open("all_ChEA_2016.txt","r")
libraryFileLines = libraryFile.readlines()
for line in libraryFileLines:
    line = line.rstrip("\n")
    fields = line.split("\t")
    term = fields[0]
    genes = fields[1:]
    libraryTerms.append(term)
    libraryGeneSets.append(genes)
    for gene in genes:
        allUniqueGenes.append(gene)
libraryFile.close()

# Retrieve Geo Microarray gene sets and terms 
GeoMicroarrayTerms = []
GeoMicroarrayGeneSets = []

GeoMicroarrayFile = open("Single_Gene_Perturbations_from_GEO_TFS_Microarray.txt","r")
GeoMicroarrayFileLines = GeoMicroarrayFile.readlines()
for line in GeoMicroarrayFileLines:
    line = line.rstrip("\n")
    fields = line.split("\t")
    term = fields[0]
    term = term.replace(",","")
    genes = fields[1:]
    GeoMicroarrayTerms.append(term)
    GeoMicroarrayGeneSets.append(genes)
    for gene in genes:
        allUniqueGenes.append(gene)
GeoMicroarrayFile.close()

# Sort all unique genes by name 
# This will be the order of genes in matrix
allUniqueGenes=np.unique(allUniqueGenes)
allUniqueGenes=np.sort(allUniqueGenes)

# Dictionary holds all unique genes and their corresponding indeces (column numbers in matrix)for quick look-up 
uniqueGenesIndexes = {}
for i, gene in enumerate(allUniqueGenes):
    uniqueGenesIndexes[gene]=i

#################################################################################################################################################

# Create matrices for library and pertubation experiments
# Rows = gene sets
# Columns= all unique genes 
def CreateMatrix(geneSets):
    allRows = []
    for geneList in geneSets:
        row = []
        for gene in allUniqueGenes:
            if gene in geneList:
                row.append(1)
            else:
                row.append(0)
        allRows.append(row)
    return(np.matrix(allRows))

# Geo Microarray sparse matrix and gene set lengths
GeoMicroarrayMatrix = CreateMatrix(GeoMicroarrayGeneSets)
GeoMicroarrayGeneSetLengths = np.sum(GeoMicroarrayMatrix,axis=1)
GeoMicroarrayMatrix = sparse.lil_matrix(GeoMicroarrayMatrix)
GeoMicroarrayMatrix=GeoMicroarrayMatrix.tocsr()

# Library (Chea or Encode) sparse matrix and gene set lengths
libraryMatrixOriginal = CreateMatrix(libraryGeneSets)
libraryGeneSetLengths = np.sum(libraryMatrixOriginal,axis=1)
libraryMatrixOriginal = sparse.lil_matrix(libraryMatrixOriginal)

#################################################################################################################################################

# Find overlap of gene sets 
# Input to function is a pertubation matrix and a library (Chea/Encode) matrix
# Overlap is calucated with matrix dot product 
# Returns matrix of overlap lengths- rows are pertubation experiment sets and columns are Chea/Encode sets
def FindOverlap(expMatrix,libMatrix):
    newMatrix = np.dot(expMatrix, np.transpose(libMatrix))
    return(newMatrix)

#################################################################################################################################################

# Function removes % of frequently occurring genes in Chip-seq
# Input is % of genes to remove, liklihood removal dictionary, and library matrix
def MatrixRemoveGenes(percent,liklihoodDictOrig,libraryMatrixOrig):
    liklihoodDict=copy.deepcopy(liklihoodDictOrig)
    libraryMatrix=copy.deepcopy(libraryMatrixOrig)
    if percent != 0:
        # Array of genes to remove
        removeGenes = [] 
        
        # Length of dictionary (keys)
        numberOccurrences = len(liklihoodDict)
        
        # Number of genes that should be removed
        numberGenesRemove = int(percent*len(ChipseqDict))
        
        # Randomly draw numbers(dictionary keys), add corresponding gene to list of genes to be removed until
        # the number of genes to be removed is reached
        while(len(removeGenes)<numberGenesRemove):
            sampleNumber = np.random.randint(1,numberOccurrences)
            gene = liklihoodDict[sampleNumber]
            if gene not in removeGenes:
                removeGenes.append(gene)
                
        # Find column indexes of genes to be removed from library matrix
        # If gene is not in matrix, continue loop...
        # It is possible some genes that occur in Chip-seq libraries may not be in Chea/Encode and pertubation sets
        # and therefore is not in matrix
        indexesRemoveGenes = []
        for gene in removeGenes:
            try:
                indexesRemoveGenes.append(uniqueGenesIndexes[gene])
            except:
                continue
        
        # Replace columns for genes that need to be removed with column of 0's
        indexesRemoveGenes=list(indexesRemoveGenes)
        libraryMatrix[:,indexesRemoveGenes]=0 
        libraryMatrix=libraryMatrix.tocsr()
        
        # Return matrix with gene columns removed 
        return(libraryMatrix)
        
    # If % of genes to remove is 0, return original library matrix
    else:
        libraryMatrix=libraryMatrix.tocsr()
        return(libraryMatrix)

#################################################################################################################################################

# Define max N and d (universe) for fast fisher's test
universe =20000
maximumN =universe+int(max(GeoMicroarrayGeneSetLengths))+int(max(libraryGeneSetLengths))

# Create vector F for fast fisher's test calculation
F = [0]
for i in range(1,maximumN+1):
    F.append(F[i-1]+np.log10(i))

# Function performs fast fisher exact test between each experiment and all library sets 
# Input is: overlap matrix, Chea/Encode library matrix (genes removed), experiment gene set lengths, experiment TF's, Chea/Encode library TF's
# Calculates p-values, finds TF matches in Chea/Encode library, records rank
# Returns dictionary containing ALL ranks as keys and occurrences as values
def FastFishersAndCalcRanks(overlapMatrix,libraryMatrixGenesRemoved,lengthsExpSets,experimentTFTerms,libraryTFTerms):
    # Dictionary holds ranks (keys) and their occurrences (values)
    RanksDict={}
    
    # Lengths of library gene sets after gene removal
    lengthsLibrarySets= np.sum(libraryMatrixGenesRemoved,axis=1) 
    
    # Experiment Transcription Factors 
    experimentTFs = []
    for term in experimentTFTerms:
        termName = term.split("_")[0]
        experimentTFs.append(termName) 
    
    # Chea/Encode Transcription Factors
    libraryTFs = []
    for term in libraryTFTerms:
        termName = term.split("_")[0]
        libraryTFs.append(termName)

    # If experiment TF is in Chea/Encode library, calculate p-value for each library set
    for expNum in range(0,len(experimentTFs)):
        if experimentTFs[expNum] in libraryTFs:
            allFisher=[]
            for libNum in range(0,len(libraryTFs)):
                a = int(overlapMatrix[expNum,libNum])
                b = int(lengthsExpSets[expNum]-a)
                c = int(lengthsLibrarySets[libNum]-a)
                n= 20000
                d=n-a-b-c 
                maxOverlap = min(b,c)
                pvalue = 10**(F[(a+c)]+F[(a+b)]+F[(c+d)]+F[(b+d)]-F[a]-F[b]-F[c]-F[d]-F[n])
                invariant = (F[a+c]+F[a+b]+F[c+d]+F[b+d]-F[n])
                for i in range(1,maxOverlap+1):
                    pvalue = pvalue+ 10**(invariant - (F[a+i]+F[b-i]+F[c-i]+F[d+i]))
                allFisher.append(pvalue)            
            
            
            # Find rank of p-value for TF matches
            indexTFName=[i for i, x in enumerate(libraryTFs) if x == experimentTFs[expNum]]
            allFisher=np.array(allFisher)
            pvalues = list(allFisher[indexTFName])
            allSortedFishers = sorted(allFisher)
            for p in pvalues:
                firstindex=allSortedFishers.index(p)
                numSamePValue = allSortedFishers.count(p)
                ranksSamePValue = list(range(firstindex+1,firstindex+1+numSamePValue))
                rank=random.choice(ranksSamePValue)
                if rank not in RanksDict.keys():
                    RanksDict[rank]=1
                else:
                    RanksDict[rank]+=1
    # Return dictionary of ranks 
    return(RanksDict)
    
#################################################################################################################################################
# Random Chea genes(5% removed) Ranks
def geneRemovalRanksResults(percentRemove,repeat,geneRemovalDict,libraryMATRIX,experimentMatrix,experimentGeneSetLengths,libraryTerms,experimentTerms):
    finalRanksDict=Counter({})
    for i in range(0,repeat):
        print(i)
        percentLibraryMatrixRemoved= MatrixRemoveGenes(percentRemove,geneRemovalDict,libraryMATRIX)
        overlap=FindOverlap(experimentMatrix,percentLibraryMatrixRemoved)
        ranksDict=FastFishersAndCalcRanks(overlap,percentLibraryMatrixRemoved,experimentGeneSetLengths,experimentTerms,libraryTerms)
        ranksDict=Counter(ranksDict)
        finalRanksDict=finalRanksDict+ranksDict

    return(finalRanksDict)

#################################################################################################################################################

# Corrected Chea/Encode: 10% genes removed based on occurrence 100 times, write ranks to a file
Corrected=geneRemovalRanksResults(0.1,100,liklihoodRemovalDict,libraryMatrixOriginal,GeoMicroarrayMatrix,GeoMicroarrayGeneSetLengths,libraryTerms,GeoMicroarrayTerms)

fileOpen = open("CorrectedChea10PercentRemovedRanks.csv","w")
fileOpen.write("Ranks"+","+"Occurrences/Counts"+"\n")
for key,value in Corrected.items():
    fileOpen.write(str(key)+","+str(value)+"\n")
fileOpen.close()


# Random Chea/Encode: 10% genes removed randomly 100 times, write ranks to a file
Random=geneRemovalRanksResults(0.1,100,equalLiklihoodRemovalDict,libraryMatrixOriginal,GeoMicroarrayMatrix,GeoMicroarrayGeneSetLengths,libraryTerms,GeoMicroarrayTerms)
    
fileOpen = open("RandomChea10PercentRemovedRanks.csv","w")
fileOpen.write("Ranks"+","+"Occurrences/Counts"+"\n")
for key,value in Random.items():
    fileOpen.write(str(key)+","+str(value)+"\n")
fileOpen.close()


# Normal Chea/Encode: no genes removed, write ranks to a file
Normal=geneRemovalRanksResults(0,1,equalLiklihoodRemovalDict,libraryMatrixOriginal,GeoMicroarrayMatrix,GeoMicroarrayGeneSetLengths,libraryTerms,GeoMicroarrayTerms)

fileOpen = open("EncodeRanks.csv","w")
fileOpen.write("Ranks"+","+"Occurrences/Counts"+"\n")
for key,value in Normal.items():
    fileOpen.write(str(key)+","+str(value)+"\n")
fileOpen.close()


"""
Created on Mon May 14 21:02:39 2018

@author: megan wojciechowicz, maayan lab 
"""

