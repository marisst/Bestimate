def save_logs(filename, epoch, logs):

    with open(filename, "a", newline="", encoding="utf-8-sig") as resultFile:
        print(",".join([str(epoch), "%.4f" % logs['loss'], "%.4f" % logs['val_loss']]), file=resultFile)