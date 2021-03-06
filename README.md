# PLATO

   PLATO is a semi-supervised learning approach for somatic variant calling and peptide identification which can  be used in personalized cancer immunotherapy. The current version provides peptide identification from MS/MS data. It uses AutoML service on Azure for modeling part.
   
   Contact: elham.sherafat@uconn.edu

# Download and installation

## Git (All the dependencies should be properly installed)

### Dependencies
python    

### Steps
1) Create an Azure account

2) Make a resource group and workspace 

3) Download the latest version of PLATO from https://github.com/esherafat/PLATO
    
    git clone https://github.com/esherafat/PLATO.git
    
4) Unzip the source code and go into the directory by using the following command:

    tar xvzf PLATO-*.tar.gz

    cd PLATO

5) Replace "YOUR_SUBSCRIPTION_ID", "YOUR_RESOURCE_GROUP", and "YOUR_WORKSPACE" at config.json file with yours

# General usage

There are 16 arguments which can be changed. Please look at the help to learn more about the arguments. 

Example: 
python3 PLATO_v20.06.py --input_folder ./data/ --output_folder ./data_result/ --sample_name human-mel3-20140304-1  --autoML_iterations 15 > ./data/human-mel3-20140304.log

# Input files

PLATO takes tsv files as input. Input folder should include two files per sample: 
sample_name-POSITIVES.txt and sample_name-ALL.txt'
 
 
