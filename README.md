# SupervisedClusteringTeleassistance

# Instructions to use the program:

- Clone the code from github
- Create a virtual environment
- Install the required libraries (you can find them in 'requirements.txt')

The program has various scripts. The most importants are 'main_organizzato.py' and 'clustering.py'.
To view the results (numerical and graphics) open 'clustering.py' and run the code.
In the end, in the terminal, you will see the numerical outputs and an image will appear. You can save that image if you want, with the save button.
Then you can close that image, and after that the following image will appear. Repeat this steps until all images are appeared.

You can also modify some settings and parameters:
- features:
  There are various features, some of them are numerical type and the other are categorical type. To run the experiment with different features,
  go in 'clustering.py', then go to the line 470 and insert in the specific space the numerical features that you want (for example 'età').
  Similary, in the line 471 insert in the specific space tha categorical features tha you want to use (for example 'tipologia_professionista_sanitario').
  You can find all the features that are available in 'main_organizzato.py' in the line 139 and following.

  - k (number of centroids):
    You can modify the value of k in 'clustering.py' in the line 467

  - number of runs:
    You can modify this value in 'clustering.py' in the line 537

  -max number of iterations:
  You can modify this value in 'clustering.py' in the line 538
  



