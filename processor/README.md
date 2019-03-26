# Proccesor

The processor module runs on an independent compute instance with GPU resources.

The responsibilities of the processor are to

1. Process raw audio data into spectrograms
2. Perform speaker embedding inference
3. Train LR for registration and do LR inference for login
4. Perform speech rec inference


## Redis Queues

We are going to maintain an in-memory redis queues to pass information 
between the processor and the webserver. The `queue:request` contains request information sent from the webserver to the processor.
The request information is a json with keys `[id, timestamp]`, where `id` is a unique identifier of the 
request

The webserver stores the audio with the key `audio:id`, which the processor can lookup using
the request information. 

The processor then stores the result as a json with keys `[id, timestamp, speaker_id]` using redis key `result:id`,
which the webserver can lookup.




Reference documentation is [here](https://redislabs.com/ebook/part-2-core-concepts/chapter-6-application-components-in-redis/6-4-task-queues/6-4-1-first-in-first-out-queues/).