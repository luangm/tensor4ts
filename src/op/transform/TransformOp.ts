import Tensor from "../../Tensor";
import Operation from "../Operation";

export default abstract class TransformOp extends Operation {

  constructor(input: Tensor, result: Tensor) {
    super(input, null, result);
  }

  exec(dim?: number): void {
    // nothing
  }

}