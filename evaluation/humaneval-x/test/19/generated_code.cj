main() {
    try{
    let correct = ArrayList<Bool>(
            sortNumbers("") == (""),
            sortNumbers("three") == ("three"),
            sortNumbers("three five nine") == ("three five nine"),
            sortNumbers("five zero four seven nine eight") == ("zero four five seven eight nine")
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