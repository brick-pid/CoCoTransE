main() {
    try{
    let correct = ArrayList<Bool>(
            sortThird(ArrayList<Int64>(5, 6, 3, 4, 8, 9, 2)) == (ArrayList<Int64>(2, 6, 3, 4, 8, 9, 5)),
            sortThird(ArrayList<Int64>(5, 8, 3, 4, 6, 9, 2)) == (ArrayList<Int64>(2, 8, 3, 4, 6, 9, 5)),
            sortThird(ArrayList<Int64>(5, 6, 9, 4, 8, 3, 2)) == (ArrayList<Int64>(2, 6, 9, 4, 8, 3, 5)),
            sortThird(ArrayList<Int64>(5, 6, 3, 4, 8, 9, 2, 1)) == (ArrayList<Int64>(2, 6, 3, 4, 8, 9, 5, 1))
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