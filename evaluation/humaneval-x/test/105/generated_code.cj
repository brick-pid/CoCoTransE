main() {
    try{
    let correct = ArrayList<Bool>(
            byLength(ArrayList<Int64>(2, 1, 1, 4, 5, 8, 2, 3))==ArrayList<String>("Eight", "Five", "Four", "Three", "Two", "Two", "One", "One" ),
            byLength(ArrayList<Int64>())==ArrayList<String>(),
            byLength(ArrayList<Int64>(1, -1, 55))==ArrayList<String>("One" ),
            byLength(ArrayList<Int64>(1, -1, 3, 2))==ArrayList<String>("Three", "Two", "One" ),
            byLength(ArrayList<Int64>(9, 4, 8))==ArrayList<String>("Nine", "Eight", "Four" )
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