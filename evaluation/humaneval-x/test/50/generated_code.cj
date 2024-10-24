var letters = "abcdefghijklmnopqrstuvwxyz".toArray();
let rand = Random(42);

public func random_string(length : Int64): String {
    var sb = StringBuilder();
    for (i in 0..length) {
        sb.append(letters[rand.nextInt64(26)]);
    }
    return sb.toString();
}
func encodeShift(s: String): String {
    var sb = StringBuilder()
    for (ch in s.toRuneArray()) {
        sb.append(Char(Int64(UInt32('a')) + ((Int64(UInt32(ch)) + Int64(UInt32(5)) - Int64(UInt32('a'))) % Int64(UInt32(26)))))
    }
    sb.toString()
}
main() {
    try{
        for (i in 0..100) {
            var str = random_string(rand.nextInt64(10) + 10);
            var encode_str = encodeShift(str);
            if (!(decodeShift(encode_str)==(str))) {
                print("false")
            }
        }
    }
    catch(e: Exception)
    {
        print("false")
    }
}