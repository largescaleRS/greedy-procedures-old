EFGPlusPlus.py includes the implementation of the parallelized EFG++ procedure. The parallelization is based on the library **multipleprocessing**. The experiment is conducted on a PC with 72 cores and .

For the experiment with random simulation times described in EC 4.3.2, we use the library **win_precise_time** to control the sleep of the processors. We do this because, on Windows systems, the time.sleep() function has an unsatisfying precision level. We note that this is no longer an issue when using time.sleep() on a Mac OS system. 
