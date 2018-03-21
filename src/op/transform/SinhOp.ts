import TransformOp from "./TransformOp";

export default class SinhOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.sinh(a);
  }

}