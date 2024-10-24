main() {
    try{
    let correct = ArrayList<Bool>(
            iscube(1),
            !iscube(2),
            iscube(-1),
            iscube(64),
            !iscube(180),
            iscube(1000),
            iscube(0),
            !iscube(1729)
    );
if (correct.contains(false)) {
                print(correct)
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}