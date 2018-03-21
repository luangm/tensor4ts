import TransformOp from "./TransformOp";

export default class ReluOp extends TransformOp {

  body(a: number, b?: number): number {
    return a > 0 ? a : 0;
  }

}