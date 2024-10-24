main() {
    try{
    let correct = ArrayList<Bool>(
            prodSigns(ArrayList<Int64>(1, 2, 2, -4)).getOrThrow() == -9,
            prodSigns(ArrayList<Int64>(0, 1)).getOrThrow() == 0,
            prodSigns(ArrayList<Int64>(1, 1, 1, 2, 3, -1, 1)).getOrThrow() == -10,
            prodSigns(ArrayList<Int64>) == None,
            prodSigns(ArrayList<Int64>(2, 4,1, 2, -1, -1, 9)).getOrThrow() == 20,
            prodSigns(ArrayList<Int64>(-1, 1, -1, 1)).getOrThrow() == 4,
            prodSigns(ArrayList<Int64>(-1, 1, 1, 1)).getOrThrow() == -4,
            prodSigns(ArrayList<Int64>(-1, 1, 1, 0)).getOrThrow() == 0
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