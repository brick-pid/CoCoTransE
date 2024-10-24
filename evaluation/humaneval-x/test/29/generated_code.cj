main() {
    try{
    let correct = ArrayList<Bool>(
            filterByPrefix(ArrayList<String>(), "john") == (ArrayList<String>()),
            filterByPrefix(ArrayList<String>("xxx", "asd", "xxy", "john doe", "xxxAAA", "xxx"), "xxx") == (ArrayList<String>("xxx", "xxxAAA", "xxx"))
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