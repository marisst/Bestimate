def save_logs(directory_name, epoch, logs):
    with open(directory_name + "/results.csv", "a", newline="", encoding="utf-8-sig") as resultFile:
        print(",".join([str(epoch), "%.4f" % logs['loss'], "%.4f" % logs['val_loss']]) + "\n", file=resultFile)
