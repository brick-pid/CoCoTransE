main() {
    try{
    let correct = ArrayList<Bool>(
            strangeSortList(ArrayList<Int64>(1, 2, 3, 4)) == (ArrayList<Int64>(1, 4, 2, 3)),
            strangeSortList(ArrayList<Int64>(5, 6, 7, 8, 9)) == (ArrayList<Int64>(5, 9, 6, 8, 7)),
            strangeSortList(ArrayList<Int64>(1, 2, 3, 4, 5)) == (ArrayList<Int64>(1, 5, 2, 4, 3)),
            strangeSortList(ArrayList<Int64>(5, 6, 7, 8, 9, 1)) == (ArrayList<Int64>(1, 9, 5, 8, 6, 7)),
            strangeSortList(ArrayList<Int64>(5, 5, 5, 5)) == (ArrayList<Int64>(5, 5, 5, 5)),
            strangeSortList(ArrayList<Int64>()) == (ArrayList<Int64>()),
            strangeSortList(ArrayList<Int64>(1, 2, 3, 4, 5, 6, 7, 8)) == (ArrayList<Int64>(1, 8, 2, 7, 3, 6, 4, 5)),
            strangeSortList(ArrayList<Int64>(0, 2, 2, 2, 5, 5, -5, -5)) == (ArrayList<Int64>(-5, 5, -5, 5, 0, 2, 2, 2)),
            strangeSortList(ArrayList<Int64>([111111])) == (ArrayList<Int64>([111111]))
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