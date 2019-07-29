module load mongodb/3.4.10
port=`findport 27017`
mongod --dbpath /scratch/achattor/mongodb_storage/trial --port $port > mongod.log &
