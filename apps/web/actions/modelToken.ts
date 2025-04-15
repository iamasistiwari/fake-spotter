import sha256 from "crypto-js/sha256";

export default function getModelToken() {
  const MODEL_SECRET = process.env.MODEL_TOKEN_SECRET;
  if (!MODEL_SECRET) {
    throw new Error("MODEL TOKEN NOT FOUND");
  }
  const indiaDateObj = new Date().toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
  });
  const indiaDate = new Date(
    indiaDateObj.replace(/(\d{1,2})\/(\d{1,2})\/(\d{4})/, "$3-$2-$1"),
  );
  const timestamp = indiaDate.getHours() + indiaDate.getDate();
  const token = sha256(timestamp + MODEL_SECRET).toString();
  return token
}

const data = getModelToken()
console.log(data)
