def save_logs(filename, epoch, logs):

    with open(filename, "a", newline="", encoding="utf-8-sig") as resultFile:
        if logs.get('val_loss') is not None:
            print(",".join([str(epoch), "%.4f" % logs['loss'], "%.4f" % logs['val_loss']]), file=resultFile)
        else:
            print(",".join([str(epoch), "%.4f" % logs['loss'], "no_val_loss"]), file=resultFile)