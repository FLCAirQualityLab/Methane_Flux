"""
AESTHETIC LOADING UI
MADE BY: ARHOLASEK
DATE:    06/06/2025
NOT TO BE USED WITHOUT PERMISSION
@mal3v0l3nce
"""

#%% Importing loading UI essentials...
import time, sys, threading

#%% Loading UI functions...
def loading(event, t):
    while not event.is_set():
        time.sleep(t)
        sys.stdout.write(".")
        sys.stdout.flush()

def startLoading(message = "", t = 0.1):
    """
    Starts loading UI
    Input:
        message = ""  —  Message to print at start of loading (if desired)
        t       = 0.1 [seconds] — default cycle timing
    Returns:
        loadingEvent, loadingThread
        ^^^ pass into stopLoading() to stop the loading UI
    """
    if message != "": print(message, end = " ")
    loadingEvent = threading.Event()
    loadingThread = threading.Thread(target=loading, args=(loadingEvent, t))
    loadingThread.daemon = True
    loadingThread.start()
    return loadingEvent, loadingThread

def stopLoading(loadingEvent, loadingThread, done=True):
    """
    Stops loading UI
    Input:
        loadingEvent, loadingThread — Outputs of startLoading()
        done = True  —  Whether or not to print "Done!" message
        vars automatically deleted after completion
    """
    loadingEvent.set()
    loadingThread.join()
    if done == True: print(" Done!")
