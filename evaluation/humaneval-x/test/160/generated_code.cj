main() {
    try{
    let correct = ArrayList<Bool>(
            doAlgebra(ArrayList<String>("**", "*", "+"), ArrayList<Int64>(2, 3, 4, 5)) == 37,
            doAlgebra(ArrayList<String>("+", "*", "-"), ArrayList<Int64>(2, 3, 4, 5)) == 9,
            doAlgebra(ArrayList<String>("/", "*"), ArrayList<Int64>(7, 3, 4)) == 8,
            doAlgebra(ArrayList<String>("+", "**", "**"), ArrayList<Int64>(7, 5, 3, 2)) == 1953132
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