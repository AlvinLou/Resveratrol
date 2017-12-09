# Resveratrol
This is a project of a small database containing the bingding proteins of Resveratrol.
## Background
In this part we will have a generall knowledge of the background of resveratrol.
### So what is Resveratrol?
Resveratrol is a stilbenoid, a type of natural phenol and a phytoalexin produced by several plants when the plant is under attack by pathogens such as bacteria or fungi. Resveratrol is widespread in our daily food such as grapes, peanuts and so on. This substance is so popular among the research area until now is mainly because Dipak Kumar Das’s fraud that shows it can help cure some disease. While it is not confirmed to have any efficacy yet.
### The research history of Resveratrol
- 1939, Michio Takaoka(Japan) first derived it from plant roots.
- 1980s, The Japanese scientists started to do research on the Antioxidant effect.
- 1990s, Dipak Kumar Das in University of Connecticut came up with a theory that the Resveratrol in Grape wine explained French Paradox.
- 2012, University of Connecticut dismissed Professor Dipak K. Das due to the fraud in the study resveratrol
> **So that is somehow the most direct reason why this substance is still so hot even today.**
## Binding protein
A binding protein is any protein that acts as an agent to bind two or more molecules together.
### DNA binding protein
- Proteins composed of DNA-binding domains 
- have a specific or general affinity for either single or double stranded DNA.
- Generally in the major groove if the binding is sequence-specific – as with transcription factors that regulate expression of genes, and nucleases that cleave DNA between nucleotides. 
- Can also bind DNA non-specifically, such as polymerases and histones. 
- About 6%~7% are DNA binding protein
## ML models
### Neural Network
- Simulate creatures’ neuron
- Accept multiple inputs with different weight and output signal to other neurons
- Better for feature learning
- Use functions like Sigmoid to simulate signal strength
### LSTM
- LSTM stands for Long Short-Term Memory
- LSTM is one of the models of Recurrent Neural Network
- Accept previous output as one of the inputs, to simulate “memory”
- Better for sequence learning
## What we do in this project
### First Part
Doing a clustering in the selected 23 Resveratrol's binding protein. clustering means devide the original data into several groups without knowing them before, it's just different from classification.
### Second Part
put 2 stream of data, ones are Resveratrol's binding proteins, the others are the DNA binding proteins. then use machine learning to learn the features of these two stream of data, and finally we can use this model to do the protein prediction.
This prediction model is to do prediction shows which kind of protein the input protein belongs to. This is meaningful because there are not a very good categorizing method until now, so if we can do a great prediction model, we can help do this.
