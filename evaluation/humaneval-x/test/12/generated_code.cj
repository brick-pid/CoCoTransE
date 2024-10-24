main() {
    try{
    let correct = ArrayList<Bool>(
            (longest(ArrayList<String>()).isNone()), 
            (longest(ArrayList<String>("x", "y", "z")).getOrThrow() == "x"),
            (longest(ArrayList<String>("x", "yyy", "zzzz", "www", "kkkk", "abc")).getOrThrow() =="zzzz")
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