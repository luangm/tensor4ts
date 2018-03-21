import TransformOp from "./TransformOp";

export default class SquareOp extends TransformOp {

  body(a: number, b?: number): number {
    return a * a;
  }

}