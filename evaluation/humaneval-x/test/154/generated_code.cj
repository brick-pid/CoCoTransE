main() {
    try{
    let correct = ArrayList<Bool>(
            cycpatternCheck("xyzw", "xyw") == false,
            cycpatternCheck("yello", "ell") == true,
            cycpatternCheck("whattup", "ptut") == false,
            cycpatternCheck("efef", "fee") == true,
            cycpatternCheck("abab", "aabb") == false,
            cycpatternCheck("winemtt", "tinem") == true
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