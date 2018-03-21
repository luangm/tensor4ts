import TransformOp from "./TransformOp";

export default class AcoshOp extends TransformOp {

  body(a: number, b?: number): number {
    return Math.acosh(a);
  }

}