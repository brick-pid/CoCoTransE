main() {
    try{
    let correct = ArrayList<Bool>(
            !isHappy("a"),
            !isHappy("aa"),
            isHappy("abcd"),
            !isHappy("aabb"),
            isHappy("adb"),
            !isHappy("xyy"),
            isHappy("iopaxpoi"),
            !isHappy("iopaxioi")
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