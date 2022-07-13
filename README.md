# Read Me: Hierachical Neural Networks Built with TensorFlow
<div id="header" align="center">
  <img src="https://media.giphy.com/media/26xBtSyoi5hUUkCEo/giphy.gif" width="500"/>
  <img src="https://media.giphy.com/media/S2azltVMwqjv56h00q/giphy.gif" width="480"/>
  
  <div id="badges"  align="center">
    <a href=[Joseph Lazarus]"https://www.linkedin.com/in/josephlazarus1/">
      <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white&label=Joseph Lazarus">
    </a>
    <a href="https://www.linkedin.com/in/rickfontenot/">
     <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white&label=Rick Fontenot"/>
    </a>
    <a href="https://www.linkedin.com/in/purirudick/">
      <img src="https://img.shields.io/badge/LinkedIn-blue?style=for-the-badge&logo=linkedin&logoColor=white&label=Puri Rudick"/>
    </a>
  <br><img src="https://komarev.com/ghpvc/?username=rickfontenot&style=flat-square&color=blue" alt=""/>
 </div>
---

### :page_with_curl: About Our Research :
We compare the Performance of a Traditional Neural Network to a Hierachical Neural Network
- :microscope: Use Open Source Library TensorFlow to compare Hierarchical Neuerl Network Performance to traditional Neural Netowrk Implementation
  
- :dress: Use Fashion-MNIST data set to benchmark our results

- :zap: Built custom wrapper for prediction pipeline that auto-generates hierarchy based on the dictionary
  
- :open_book: Full Research Paper Available on SMU Data Science Journal


- :mailbox:How to reach us: [![Linkedin Badge](https://img.shields.io/badge/-Lazarus-blue?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/josephlazarus1) * [![Linkedin Badge](https://img.shields.io/badge/-Rudick-blue?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/purirudick) * [![Linkedin Badge](https://img.shields.io/badge/-Fontenot-blue?style=flat&logo=Linkedin&logoColor=white)](https://www.linkedin.com/in/rickfontenot)

- Navigating the code:
  -- The notebook "HNN_manually" is a long form construction of an HNN using the first hierarchy. The advantage is each submodel can be tuned differently, the down side is it takes a lot of time and attention to change to a different hierarchical structure
  -- The Model 1 & Model 3 folders are traditional flat NN models for the purpose of comparison in our study
  -- The Model 4A, 4B, and 4C folders contain HNN models with different structures. The dictionary defines the hierarchy structure and can be changed to easily assess other structures as the models prediction pipelines are auto-generated. The down side is it's not easy to tune each submodel differently. Once a good hierarchy structure is identified, it could be implemented similar to the long form code in HNN_manually if you want to tune each submodel differently
---


  
