errors = ["this", "is", "a", "test"]
if(len(errors) > 0):
    with open('./errors.txt', 'w') as f:
        for e in errors:
            f.write("%s\n" % e)