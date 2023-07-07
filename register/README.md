# Objects and Subjects

* **objects.csv**: the list of used objects. Each line is an object. Each header is separated by TAB i.e. '\t'.
  * object name
  * characteristic:
    * rigid
    * soft
    * printed
  * attached with optitrack markers
    * 1: yes
    * 0: no
* **subjects.csv**: the list of the subjects participating in the experiments
  * subject ID

## Add new subject or object

It can be easily done by appending a new line at the end of the file for the new object or subject. There are some cautions about adding new objects:

* the new object must be appended at the end of the list so that the IDs of the existing object will not be affected.
* three descriptions (headers) of an object is separated by TAB ('\t') instead of SPACE (' ').