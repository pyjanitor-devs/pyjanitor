docker-compose commands
ref email: deploying <name-withheld> docker for your <project-name>
1. copy http:<devBase>/dist/dockerSeasons/docker directory 
   in each of your <project-name>s directories.
2. copy http:<devBase>/dist/dockerSeasons/docker-compose.yaml file 
   in each of your <project-name>s directories.
3. symbolically link your  <project-name>s requirements.txt 
   to /docker/requirements.txt
   (goto <project-name>/docker);
   $ ln ../requirements.txt requirements.txt
ref: https://pip.pypa.io/en/stable/user_guide/#requirements-files

4. compose all containers (services) into one image
   $ cd to docker-compose.yaml directory;
   $ docker-compose build

5.runs docker-compose image in background
   $ docker-compose up & (add to <home>/<uid>/.prolile)
 note:
   $ docker-compose up -d
   will NOT output 
 http://127.0.0.1:8888/?token=a5bcd81e7c5b126cb1b67ddf7fa6ce4a49eff7cef9f48ad (example)
6.shuts down docker-compose image in background
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
devdir='<your-local-path>/PROJECTS/dockerSeasons/Dev'
testdir='<your-local-path>/Documents/PROJECTS/dockerSeasons/Test'

#
alias updev="cd $devdir; docker-compose up &"
alias downdev="cd $devdir; docker-compose down" 
alias builddev="cd $devdir; docker-compose build"

#
alias uptest="cd $testdir; docker-compose up &"
alias downtest="cd $testdir; docker-compose down"
alias buildtest="cd $testdir; docker-compose build"


