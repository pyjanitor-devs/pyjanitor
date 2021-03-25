ref: Adding Jupyter Notebook Extensions to a Docker Image
https://towardsdatascience.com/adding-jupyter-notebook-extensions-to-a-docker-image-851bc2601ca3


1. compose all containers (services) into one image
   $ cd to docker-compose.yaml directory;
   $ docker-compose build

2.runs docker-compose image in background
   $ docker-compose up & (add to <home>/<uid>/.prolile)
 note:
3.shuts down docker-compose image in background
   $ docker-compose down

note: image is a running container

 show all docker images 
 $ docker ps -a
 show all container that can make images
  $ docker images

stops all docker images
###docker stop $(docker ps -a -q)
removes all stopped docker images
###docker rm $(docker ps -a -q)


 delete containers and images
 - all stopped containers
 - all networks not used by at least one container
 - all dangling images
 - all dangling build cache
$ docker system prune
and all unused images 
$ docker system prune -a

if using Pycharm
ref: https://www.jetbrains.com/help/pycharm/using-docker-as-a-remote-interpreter.html
#
if you want add to .bashrc_profile or bashrc.txt
devdir='<your-local-path>/docker/dev'
testdir='<your-local-path>/docker/test

#
alias updev="cd $devdir; docker-compose up &" 
alias downdev="cd $devdir; docker-compose down"
alias builddev="cd $devdir; docker-compose build"

#
alias uptest="cd $testdir; docker-compose up &"
alias downtest="cd $testdir; docker-compose down"
alias buildtest="cd $testdir; docker-compose build"


