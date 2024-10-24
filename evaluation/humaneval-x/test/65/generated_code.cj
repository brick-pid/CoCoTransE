main() {
    try{
    let correct = ArrayList<Bool>(
            circularShift(100, 2) == ("001"),
            circularShift(12, 2) == ("12"),
            circularShift(97, 8) == ("79"),
            circularShift(12, 1) == ("21"),
            circularShift(11, 101) == ("11")
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