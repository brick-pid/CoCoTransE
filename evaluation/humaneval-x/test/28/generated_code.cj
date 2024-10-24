main() {
    try{
    let correct = ArrayList<Bool>(
            (concatenate(ArrayList<String>()) == ""),
            (concatenate(ArrayList<String>("x", "y", "z")) == "xyz"),
            (concatenate(ArrayList<String>("x", "y", "z", "w", "k")) == "xyzwk")
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